import numpy as np
import torch

dtype = torch.float64
R = 8314.46261815324  # J/kmol/K


class Gas1D:
    def __init__(self, gas):
        self.gas = gas
        self.species = gas.species()
        self.reactions = gas.reactions()
        self.n_spe = self.gas.n_species
        self.n_rxn = self.gas.n_reactions
        self.Ws = torch.tensor(gas.molecular_weights, dtype=dtype)  # kg/kmol, (n_spe, )

        # thermodynamic property polynomial coefficients
        coeffs15 = np.array([self.species[k].thermo.coeffs for k in range(self.n_spe)])  # (n_spe, 15)
        self.coeffs15 = torch.tensor(coeffs15, dtype=dtype)  # (n_spe, 15)

        # transport property polynomial coefficients
        coeffs_mu = np.array([gas.get_viscosity_polynomial(k) for k in range(self.n_spe)])  # (n_spe, 5)
        coeffs_lam = np.array([gas.get_thermal_conductivity_polynomial(k) for k in range(self.n_spe)])  # (n_spe, 5)
        coeffs_Dkj = []
        for k in range(self.n_spe):
            coeffs_Dkj.append([gas.get_binary_diff_coeffs_polynomial(k, j) for j in range(self.n_spe)])
        coeffs_Dkj = np.array(coeffs_Dkj)  # (n_spe, n_spe, 5)
        self.coeffs_mu = torch.tensor(coeffs_mu, dtype=dtype)  # (n_spe, 5)
        self.coeffs_lam = torch.tensor(coeffs_lam, dtype=dtype)  # (n_spe, 5)
        self.coeffs_Dkj = torch.tensor(coeffs_Dkj, dtype=dtype)  # (n_spe, n_spe, 5)

        # reaction information
        self.reaction_types = [rxn.reaction_type for rxn in gas.reactions()]
        self.is_2body = torch.tensor(list(map(lambda s: True if s == "reaction" else False, self.reaction_types)))
        self.is_3body = torch.tensor(list(map(lambda s: True if s == "three-body" else False, self.reaction_types)))
        self.is_falloff = torch.tensor(list(map(lambda s: True if s == "falloff" else False, self.reaction_types)))
        self.rate_types = [rxn.rate.type for rxn in gas.reactions()]
        self.is_Arrh = torch.tensor(list(map(
            lambda s: True if s == "Arrhenius" else False, self.rate_types)))  # = ~is_falloff
        self.is_Troe = torch.tensor(list(map(lambda s: True if s == "Troe" else False, self.rate_types)))
        self.is_reversible = torch.tensor([rxn.reversible for rxn in gas.reactions()])

        # stoichiometric coefficients
        self.nu_reactants = torch.tensor(gas.reactant_stoich_coeffs3, dtype=dtype)  # (n_spe, n_rxn)
        self.nu_products = torch.tensor(gas.product_stoich_coeffs3, dtype=dtype)  # (n_spe, n_rxn)
        self.nu = self.nu_products - self.nu_reactants  # (n_spe, n_rxn)
        self.nu_sum = torch.sum(self.nu, dim=0)  # (n_rxn, )

        # reaction orders (only valid for irreversible reactions)
        self.reaction_orders = self.nu_reactants.clone()  # (n_spe, n_rxn)
        for k in range(self.n_rxn):
            if not self.is_reversible[k] and gas.reaction(k).orders != {}:
                for (name, order) in gas.reaction(k).orders.items():
                    self.reaction_orders[gas.species_index(name), k] = order

        # efficiencies for calculating [M]
        efficiencies_all = []
        for i in range(self.n_rxn):
            rxn = gas.reaction(i)
            if rxn.reaction_type in ["three-body", "falloff"]:
                efficiencies = np.array([rxn.efficiency(s) for s in gas.species_names])
                efficiencies_all.append(efficiencies)
            else:
                efficiencies_all.append(np.ones(self.n_spe) * np.nan)
        self.efficiencies_all = torch.tensor(np.array(efficiencies_all), dtype=dtype)  # (n_rxn, n_spe)

        # coefficients for calculating k_f
        AbEs, AbE_ls = [], []
        Troe_coeffs = []
        for rxn in gas.reactions():
            if rxn.rate.type == "Arrhenius":
                AbEs.append([rxn.rate.pre_exponential_factor,
                             rxn.rate.temperature_exponent,
                             rxn.rate.activation_energy])
                AbE_ls.append([np.nan, np.nan, np.nan])
            else:
                AbEs.append([rxn.rate.high_rate.pre_exponential_factor,
                             rxn.rate.high_rate.temperature_exponent,
                             rxn.rate.high_rate.activation_energy])
                AbE_ls.append([rxn.rate.low_rate.pre_exponential_factor,
                               rxn.rate.low_rate.temperature_exponent,
                               rxn.rate.low_rate.activation_energy])
            if rxn.rate.type == "Troe":
                Troe_coeffs.append(rxn.rate.falloff_coeffs)
            else:
                Troe_coeffs.append([np.nan, np.nan, np.nan, np.nan])
        self.AbEs = torch.tensor(AbEs, dtype=dtype)  # (n_rxn, 3)
        self.AbE_ls = torch.tensor(AbE_ls, dtype=dtype)  # (n_rxn, 3)
        self.Troe_coeffs = torch.tensor(np.array(Troe_coeffs), dtype=dtype)  # (n_rxn, 4)

    def cal_W(self, Ys):
        """Calculate mean molecular weight. Unit: kg/kmol. Shape: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return 1 / torch.matmul(Ys, 1 / self.Ws)[:, None]

    def cal_rho(self, T, Ys):
        """Calculate mixture density. Unit: kg/m3. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return self.gas.P * self.cal_W(Ys) / (R * T)

    def cal_Xs(self, Ys):
        """Calculate species mole fractions. Shape: (n_pnt, n_spe). Ys: (n_pnt, n_spe)."""
        return Ys / self.Ws * self.cal_W(Ys)

    def cal_cs(self, T, Ys):
        """Calculate species molar concentrations. Unit: kmol/m3. Shape: (n_pnt, n_spe).
         T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return Ys / self.Ws * self.cal_rho(T, Ys)

    def get_coeffs7(self, T):
        """Get the 7 coefficients of thermodynamics property polynomial. Shape: (n_pnt, n_spe, 7). T: (n_pnt, 1)."""
        T_mediums = self.coeffs15[:, 0]  # (n_spe, )
        is_bigger = T > T_mediums  # (n_pnt, n_spe)
        is_bigger_rep = is_bigger[:, :, None].repeat(1, 1, 7)  # (n_pnt, n_spe, 7)
        coeffs15_rep = self.coeffs15.repeat(T.shape[0], 1, 1)  # (n_pnt, n_spe, 15)
        return torch.where(is_bigger_rep, coeffs15_rep[:, :, 1: 8], coeffs15_rep[:, :, 8: 15])  # (n_pnt, n_spe, 7)

    def cal_cps_mole(self, T):
        """Calculate species molar heat capacities. Unit: J/kmol/K. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        coeffs7 = self.get_coeffs7(T)
        coeffs = coeffs7[:, :, 0: 5]  # (n_pnt, n_spe, 5)
        T_trans = torch.hstack([torch.ones_like(T), T, T ** 2, T ** 3, T ** 4])  # (n_pnt, 5)
        return torch.einsum("...ij,...j->...i", coeffs, T_trans) * R  # (n_pnt, n_spe)

    def cal_hs_mole(self, T):
        """Calculate species molar enthalpies. Unit: J/kmol. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        coeffs7 = self.get_coeffs7(T)
        coeffs = coeffs7[:, :, 0: 6]  # (n_pnt, n_spe, 6)
        T_trans = torch.hstack([T, T ** 2 / 2, T ** 3 / 3, T ** 4 / 4, T ** 5 / 5, torch.ones_like(T)])  # (n_pnt, 6)
        return torch.einsum("...ij,...j->...i", coeffs, T_trans) * R  # (n_pnt, n_spe)

    def cal_ss_mole(self, T):
        """Calculate species molar entropies. Unit: J/kmol/K. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        coeffs7 = self.get_coeffs7(T)
        coeffs = coeffs7[:, :, [0, 1, 2, 3, 4, 6]]  # (n_pnt, n_spe, 6)
        T_trans = torch.hstack([torch.log(T), T, T ** 2 / 2, T ** 3 / 3, T ** 4 / 4, torch.ones_like(T)])  # (n_pnt, 6)
        ss_mole = torch.einsum("...ij,...j->...i", coeffs, T_trans) * R  # (n_pnt, n_spe)
        p_rel = torch.tensor(self.gas.P / self.gas.reference_pressure, dtype=dtype)
        return ss_mole - torch.log(p_rel) * R

    def cal_cps(self, T):
        """Calculate species heat capacities. Unit: J/kg/K. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        return self.cal_cps_mole(T) / self.Ws

    def cal_hs(self, T):
        """Calculate species enthalpies. Unit: J/kg. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        return self.cal_hs_mole(T) / self.Ws

    def cal_ss(self, T):
        """Calculate species entropies. Unit: J/kg/K. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        return self.cal_ss_mole(T) / self.Ws

    def cal_cp(self, T, Ys):
        """Calculate mixture heat capacity. Unit: J/kg/K. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return torch.sum(self.cal_cps(T) * Ys, dim=-1)[:, None]

    def cal_h(self, T, Ys):
        """Calculate mixture enthalpy. Unit: J/kg. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return torch.sum(self.cal_hs(T) * Ys, dim=-1)[:, None]

    def cal_s(self, T, Ys):
        """Calculate mixture entropy. Unit: J/kg/K. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Xs = self.cal_Xs(Ys)  # (n_pnt, n_spe)
        Xs_ = torch.where(Xs >= 0, Xs, torch.zeros(1, dtype=dtype))
        ss_mole = self.cal_ss_mole(T)  # (n_pnt, n_spe)
        # s_mole = torch.sum((ss_mole - R * torch.log(Xs_)) * Xs_, dim=-1)[:, None]  # (n_pnt, 1)
        s_mole = torch.sum((ss_mole * Xs_ - R * torch.log(Xs_ ** Xs_)), dim=-1)[:, None]  # (n_pnt, 1)
        return s_mole / self.cal_W(Ys)

    def cal_mus(self, T):
        """Calculate species viscosities. Unit: Pa-s. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        lnT = torch.log(T)
        lnT_pows = torch.hstack([torch.ones_like(lnT), lnT, lnT ** 2, lnT ** 3, lnT ** 4])  # (n_pnt, 5)
        return (torch.einsum("ik,jk->ij", lnT_pows, self.coeffs_mu) * T ** 0.25) ** 2

    def cal_lams(self, T):
        """Calculate species thermal conductivities. Unit: W/m/K. Shape: (n_pnt, n_spe). T: (n_pnt, 1)."""
        lnT = torch.log(T)
        lnT_pows = torch.hstack([torch.ones_like(lnT), lnT, lnT ** 2, lnT ** 3, lnT ** 4])  # (n_pnt, 5)
        return torch.einsum("ik,jk->ij", lnT_pows, self.coeffs_lam) * T ** 0.5

    def cal_Dkjs(self, T):
        """Calculate species binary diffusion coefficients. Unit: m^2/s. Shape: (n_pnt, n_spe, n_spe). T: (n_pnt, 1)."""
        lnT = torch.log(T)
        lnT_pows = torch.hstack([torch.ones_like(lnT), lnT, lnT ** 2, lnT ** 3, lnT ** 4])  # (n_pnt, 5)
        return torch.einsum("ik,...k->i...", lnT_pows, self.coeffs_Dkj) * T[:, None] ** 1.5 / self.gas.P

    def cal_mu(self, T, Ys):
        """Calculate mixture viscosity. Unit: Pa-s. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Xs = self.cal_Xs(Ys)  # (n_pnt, n_spe)
        mus = self.cal_mus(T)  # (n_pnt, n_spe)
        Wk_Wj = torch.outer(self.Ws, 1 / self.Ws)  # (n_spe, n_spe)
        muk_muj = torch.einsum("...i,...j->...ij", mus, 1 / mus)  # (n_pnt, n_spe, n_spe)
        phi_kj = 8 ** (-0.5) * (1 + Wk_Wj) ** (-0.5) * (1 + muk_muj ** 0.5 * Wk_Wj ** (-0.25)) ** 2
        return torch.sum(Xs * mus / torch.einsum("...ij,...j->...i", phi_kj, Xs), dim=-1)[:, None]

    def cal_lam(self, T, Ys):
        """Calculate mixture thermal conductivity. Unit: W/m/K. Shape: (n_pnt, 1).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Xs = self.cal_Xs(Ys)  # (n_pnt, n_spe)
        lams = self.cal_lams(T)  # (n_pnt, n_spe)
        return 0.5 * (torch.sum(Xs * lams, dim=-1) + 1.0 / torch.sum(Xs / lams, dim=-1))[:, None]

    def cal_Dkms_apo(self, T, Ys):
        """Calculate mixture binary diffusion coefficients (D'_km. apo: apostrophe). Unit: m^2/s.
        Shape: (n_pnt, n_spe). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Xs = self.cal_Xs(Ys)  # (n_pnt, n_spe)
        Dkjs = self.cal_Dkjs(T)  # (n_pnt, n_spe, n_spe)
        Dkjs_reci_0diag = 1 / Dkjs
        Dkjs_reci_0diag[:, np.arange(self.n_spe), np.arange(self.n_spe)] = 0.0
        return (1 - Ys) / torch.einsum("...ij,...j->...i", Dkjs_reci_0diag, Xs)

    def cal_delta_h0_mol(self, T):
        """Calculate reactions delta standard-state enthalpies. Unit: J/kmol. Shape: (n_pnt, n_rxn). T: (n_pnt, 1)."""
        hs_mole = self.cal_hs_mole(T)  # (n_pnt, n_spe)
        return torch.einsum("...i,ij->...j", hs_mole, self.nu)

    def cal_delta_s0_mol(self, T):
        """Calculate reactions delta standard-state entropies. Unit: J/kmol. Shape: (n_pnt, n_rxn). T: (n_pnt, 1)."""
        ss_mole = self.cal_ss_mole(T)  # (n_pnt, n_spe)
        return torch.einsum("...i,ij->...j", ss_mole, self.nu)

    def cal_Kc(self, T):
        """Calculate reactions equilibrium constants in concentration units. Shape: (n_pnt, n_rxn). T: (n_pnt, 1)."""
        delta_h0_mol = self.cal_delta_h0_mol(T)  # (n_pnt, n_rxn)
        delta_s0_mol = self.cal_delta_s0_mol(T)  # (n_pnt, n_rxn)
        Kp = torch.exp((delta_s0_mol - delta_h0_mol / T) / R)
        # return Kp * (self.gas.reference_pressure / R / T) ** self.nu_sum
        return Kp * (self.gas.P / R / T) ** self.nu_sum

    def cal_c_M(self, T, Ys):
        """Calculate reactions [M]. Unit: kmol/m3. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        cs = self.cal_cs(T, Ys)  # (n_pnt, n_spe)
        is_3orfall = torch.logical_or(self.is_3body, self.is_falloff)  # (n_rxn, )
        c_M = torch.ones([T.shape[0], self.n_rxn], dtype=dtype) * torch.nan  # (n_pnt, n_rxn)
        c_M[:, is_3orfall] = torch.einsum("ik,jk->ij", cs, self.efficiencies_all[is_3orfall, :])
        return c_M

    def cal_c_M_equ(self, T, Ys):
        """Calculate reactions equivalent [M] for RoP calculation. Unit: kmol/m3. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        cs = self.cal_cs(T, Ys)  # (n_pnt, n_spe)
        c_M_equ = torch.ones([T.shape[0], self.n_rxn], dtype=dtype)  # (n_pnt, n_rxn)
        c_M_equ[:, self.is_3body] = torch.einsum("ik,jk->ij", cs, self.efficiencies_all[self.is_3body, :])
        return c_M_equ

    def cal_k_f(self, T, Ys):
        """Calculate reactions forward rate constants. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        k_h_cal = self.AbEs[:, 0] * T ** self.AbEs[:, 1] * torch.exp(- self.AbEs[:, 2] / R / T)  # (n_pnt, n_rxn)
        k_l_cal = self.AbE_ls[:, 0] * T ** self.AbE_ls[:, 1] * torch.exp(- self.AbE_ls[:, 2] / R / T)

        Prs = k_l_cal / k_h_cal * self.cal_c_M(T, Ys)  # (n_pnt, n_rxn)
        Pr1Prs_equ = torch.ones([T.shape[0], self.n_rxn], dtype=dtype)  # equivalent (Pr / (1 + Pr)) for each reaction
        Pr1Prs_equ[:, ~self.is_Arrh] = Prs[:, ~self.is_Arrh] / (1 + Prs[:, ~self.is_Arrh])

        Troes = self.Troe_coeffs  # (n_rxn, 4)
        Fcents = (1 - Troes[:, 0]) * torch.exp(-T / Troes[:, 1]) \
                 + Troes[:, 0] * torch.exp(-T / Troes[:, 2]) + torch.exp(-Troes[:, 3] / T)  # (n_pnt, n_rxn)
        coeffs_c = -0.4 - 0.67 * torch.log10(Fcents)
        coeffs_n = 0.75 - 1.27 * torch.log10(Fcents)
        f1s = (torch.log10(Prs) + coeffs_c) / (coeffs_n - 0.14 * (torch.log10(Prs) + coeffs_c))
        Fs = 10 ** (torch.log10(Fcents) / (1 + f1s ** 2))
        Fs_equ = torch.ones([T.shape[0], self.n_rxn], dtype=dtype)  # equivalent falloff function F for each reaction
        Fs_equ[:, self.is_Troe] = Fs[:, self.is_Troe]

        return k_h_cal * Pr1Prs_equ * Fs_equ

    def cal_k_r(self, T, Ys):
        """Calculate reactions reverse rate constants. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Kc = self.cal_Kc(T)  # (n_pnt, n_rxn)
        k_f = self.cal_k_f(T, Ys)  # (n_pnt, n_rxn)
        k_r = torch.zeros([T.shape[0], self.n_rxn], dtype=dtype)  # (n_pnt, n_rxn)
        k_r[:, self.is_reversible] = k_f[:, self.is_reversible] / Kc[:, self.is_reversible]
        return k_r

    def cal_RoP_f(self, T, Ys):
        """Calculate reactions forward rates of progress. Unit: kmol/m^3/s. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        cs = self.cal_cs(T, Ys)  # (n_pnt, n_spe)
        k_f = self.cal_k_f(T, Ys)  # (n_pnt, n_rxn)
        c_M_equ = self.cal_c_M_equ(T, Ys)  # (n_pnt, n_rxn)
        # return k_f * torch.prod(cs[:, :, None] ** self.nu_reactants, dim=-2) * c_M_equ  # (n_pnt, n_rxn), bug: 0 ** 0
        rop_f = torch.zeros([T.shape[0], self.n_rxn], dtype=dtype)
        idx = self.reaction_orders != 0  # (n_spe, n_rxn)
        cs_ = torch.where(cs >= 0, cs, torch.zeros(1, dtype=dtype))
        for i in range(self.n_rxn):
            rop_f[:, i] = torch.prod(cs_[:, idx[:, i]] ** self.reaction_orders[idx[:, i], i], dim=-1)
        rop_f *= k_f * c_M_equ
        return rop_f

    def cal_RoP_r(self, T, Ys):
        """Calculate reactions reverse rates of progress. Unit: kmol/m^3/s. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        cs = self.cal_cs(T, Ys)  # (n_pnt, n_spe)
        k_r = self.cal_k_r(T, Ys)  # (n_pnt, n_rxn)
        c_M_equ = self.cal_c_M_equ(T, Ys)  # (n_pnt, n_rxn)
        # return k_r * torch.prod(cs[:, :, None] ** self.nu_products, dim=-2) * c_M_equ  # (n_pnt, n_rxn), bug: 0 ** 0
        rop_r = torch.zeros([T.shape[0], self.n_rxn], dtype=dtype)
        idx = self.nu_products != 0  # (n_spe, n_rxn)
        for i in range(self.n_rxn):
            rop_r[:, i] = torch.prod(cs[:, idx[:, i]] ** self.nu_products[idx[:, i], i], dim=-1)
        rop_r *= k_r * c_M_equ
        return rop_r

    def cal_RoP(self, T, Ys):
        """Calculate reactions net rates of progress. Unit: kmol/m^3/s. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return self.cal_RoP_f(T, Ys) - self.cal_RoP_r(T, Ys)

    def cal_omega_dot(self, T, Ys):
        """Calculate species molar net production rates. Unit: kmol/m^3/s. Shape: (n_pnt, n_spe).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return torch.einsum("...j,ij->...i", self.cal_RoP(T, Ys), self.nu)

    def cal_omega_dot_mass(self, T, Ys):
        """Calculate species net production rates. Unit: kg/m^3/s. Shape: (n_pnt, n_spe).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return self.cal_omega_dot(T, Ys) * self.Ws

    def cal_omegaT_dot(self, T, Ys):
        """Calculate the heat release rate. Unit: W/m^3. Shape: (n_pnt, 1). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        return -torch.sum(self.cal_hs_mole(T) * self.cal_omega_dot(T, Ys), dim=-1)[:, None]


class Gas1D_1stepIr(Gas1D):
    def __init__(self, gas):
        super().__init__(gas)
        assert self.n_rxn == 1

    def cal_c_M(self, T, Ys):
        """Calculate reactions [M]. Unit: kmol/m3. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        c_M = torch.ones([T.shape[0], self.n_rxn], dtype=dtype) * torch.nan  # (n_pnt, n_rxn = 1)
        return c_M

    def cal_c_M_equ(self, T, Ys):
        """Calculate reactions equivalent [M] for RoP calculation. Unit: kmol/m3. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        c_M_equ = torch.ones([T.shape[0], self.n_rxn], dtype=dtype)  # (n_pnt, n_rxn = 1)
        return c_M_equ

    def cal_k_f(self, T, Ys):
        """Calculate reactions forward rate constants. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        k_h_cal = self.AbEs[:, 0] * T ** self.AbEs[:, 1] * torch.exp(- self.AbEs[:, 2] / R / T)  # (n_pnt, n_rxn = 1)
        return k_h_cal

    def cal_k_r(self, T, Ys):
        """Calculate reactions reverse rate constants. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        k_r = torch.zeros([T.shape[0], self.n_rxn], dtype=dtype)  # (n_pnt, n_rxn = 1)
        return k_r

    def cal_RoP_f(self, T, Ys):
        """Calculate reactions forward rates of progress. Unit: kmol/m^3/s. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        cs = self.cal_cs(T, Ys)  # (n_pnt, n_spe)
        k_f = self.cal_k_f(T, Ys)  # (n_pnt, n_rxn = 1)
        # return k_f * torch.prod(cs[:, :, None] ** self.nu_reactants, dim=-2)  # (n_pnt, n_rxn = 1), bug: 0 ** 0
        cs_ = torch.where(cs >= 0, cs, torch.zeros(1, dtype=dtype))
        return k_f * torch.prod(cs_[:, :, None] ** self.reaction_orders, dim=-2)  # (n_pnt, n_rxn)

    def cal_RoP_r(self, T, Ys):
        """Calculate reactions reverse rates of progress. Unit: kmol/m^3/s. Shape: (n_pnt, n_rxn).
        T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        rop_r = torch.zeros([T.shape[0], self.n_rxn], dtype=dtype)
        return rop_r


class Gas1D_1stepIr_Ea(Gas1D_1stepIr):
    def __init__(self, gas, arg_Ea):
        super().__init__(gas)
        self.Eas = arg_Ea[0]
        self.scale_Ea = arg_Ea[1]

    def cal_k_f(self, T, Ys):
        """Calculate reactions forward rate constants. Shape: (n_pnt, n_rxn). T: (n_pnt, 1). Ys: (n_pnt, n_spe)."""
        Ea = self.Eas / self.scale_Ea
        k_h_cal = self.AbEs[:, 0] * T ** self.AbEs[:, 1] * torch.exp(- Ea / R / T)  # (n_pnt, n_rxn = 1)
        return k_h_cal
