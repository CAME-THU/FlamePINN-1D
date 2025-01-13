"""Reference solution of 1D counterflow premixed flames based on detailed physical models."""
import cantera as ct
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 22
set_dpi = 200
plt.rcParams["font.size"] = set_fs  # default font size
plt.rcParams["font.sans-serif"] = "Arial"  # default font
# plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text

# use_legacy = True  # rate constants is multiplied by [M]
use_legacy = False  # rate constants is not multiplied by [M]
ct.use_legacy_rate_constants(use_legacy)


# mech_name = "gri30"
mech_name = "1S_CH4_MP1"
fuel, oxidizer = "CH4:1.0", "O2:1.0, N2:3.76"
mdot_u, mdot_b = 1, 1  # kg/m^2/s

gas = ct.Solution(f"{mech_name}.yaml")
case_name = mech_name

phi = 0.8
p = ct.one_atm * 1
T_u = 300.0
T_b = 300.0

length = 0.02

# save_dir = f"results/{case_name}/"
save_dir = "results/{:s}/p{:.2f}_phi{:.2f}/".format(case_name, p/ct.one_atm, phi)
os.makedirs(save_dir + "data/", exist_ok=True)
os.makedirs(save_dir + "pics/", exist_ok=True)

gas.TP = T_u, p
gas.set_equivalence_ratio(phi, fuel, oxidizer)
# gas.transport_model = "Mix"

# sim = ct.CounterflowPremixedFlame(gas, width=length)
sim = ct.CounterflowPremixedFlame(gas, grid=np.linspace(0, length, 501))

sim.reactants.mdot = mdot_u
sim.reactants.T = T_u
sim.reactants.X = gas.X
sim.products.mdot = mdot_b
sim.products.T = T_b
sim.products.X = "N2:1"

sim.boundary_emissivities = 0.0, 0.0
sim.radiation_enabled = False

# sim.set_initial_guess()
sim.set_initial_guess(equilibrate=False)
sim.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04)
loglevel = 1
# sim.solve(loglevel, auto=True)
sim.solve(loglevel, refine_grid=False)

# ----------------------------------------------------------------------
# save data
np.save(save_dir + "data/x.npy", sim.grid)
np.save(save_dir + "data/u.npy", sim.velocity)
np.save(save_dir + "data/V.npy", sim.spread_rate)
np.save(save_dir + "data/T.npy", sim.T)
np.save(save_dir + "data/Ys.npy", sim.Y.T)
np.save(save_dir + "data/rho.npy", sim.density)
np.save(save_dir + "data/pCurv.npy", sim.L[0])  # pressure curvature

# ----------------------------------------------------------------------
# plot the fields
plots = [sim.velocity, sim.spread_rate, sim.T, sim.density]
mathnames = ["$u$", "$V$", "$T$", r"$\rho$", ]
textnames = ["u", "V", "T", "rho", ]
units = ["m/s", "s$^{-1}$", "K", "kg/m$^3$", ]

for i in range(len(plots)):
    plt.figure(figsize=(8, 6))
    plt.title(f"{mathnames[i]}", fontsize="medium")
    plt.xlabel("$x$/m")
    plt.ylabel(f"{units[i]}")
    plt.plot(sim.grid, plots[i], lw=2)
    plt.savefig(save_dir + f"pics/field_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

plt.figure(figsize=(8, 6))
plt.title("Mole Fraction", fontsize="medium")
plt.xlabel("$x$/m")
for i in range(gas.n_species):
    plt.plot(sim.grid, sim.X[i], lw=2, label=f"{gas.kinetics_species_names[i]}")
    # if gas.kinetics_species_names[i] not in ["AR", "N2"]:
    #     plt.plot(sim.grid, sim.X[i], lw=2, label=f"{gas.kinetics_species_names[i]}")
plt.legend(fontsize="small")
plt.savefig(save_dir + "pics/field_Xs.png", bbox_inches="tight", dpi=set_dpi)
plt.close()

plt.figure(figsize=(8, 6))
plt.title("Mass Fraction", fontsize="medium")
plt.xlabel("$x$/m")
for i in range(gas.n_species):
    plt.plot(sim.grid, sim.Y[i], lw=2, label=f"{gas.kinetics_species_names[i]}")
    # if gas.kinetics_species_names[i] not in ["AR", "N2"]:
    #     plt.plot(sim.grid, sim.Y[i], lw=2, label=f"{gas.kinetics_species_names[i]}")
plt.legend(fontsize="small")
plt.savefig(save_dir + "pics/field_Ys.png", bbox_inches="tight", dpi=set_dpi)
plt.close()

