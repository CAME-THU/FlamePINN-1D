"""Reference solution of 1D freely-propagating premixed flames based on detailed physical models."""
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

gas = ct.Solution(f"{mech_name}.yaml")
case_name = mech_name

phi = 0.8

p, length = ct.one_atm * 1, 0.0025  # initial length
# p, length = ct.one_atm * 5, 0.001  # initial length

T_in = 300.0

# save_dir = f"results/{case_name}/"
save_dir = "results/{:s}/p{:.2f}_phi{:.2f}/".format(case_name, p/ct.one_atm, phi)
os.makedirs(save_dir + "data/", exist_ok=True)
os.makedirs(save_dir + "pics/", exist_ok=True)

gas.TP = T_in, p
gas.set_equivalence_ratio(phi, fuel, oxidizer)
# gas.transport_model = "Mix"

# sim = ct.FreeFlame(gas, width=length)
sim = ct.FreeFlame(gas, grid=np.linspace(0, length, 501))

sim.set_initial_guess()
sim.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
loglevel = 1
sim.solve(loglevel, auto=True)
# sim.solve(loglevel, refine_grid=False)

print("sL = ", sim.velocity[0])

# ----------------------------------------------------------------------
# save data
np.save(save_dir + "data/x.npy", sim.grid)
np.save(save_dir + "data/T.npy", sim.T)
np.save(save_dir + "data/Ys.npy", sim.Y.T)
np.save(save_dir + "data/u.npy", sim.velocity)
np.save(save_dir + "data/rho.npy", sim.density)
np.save(save_dir + "data/sL.npy", sim.velocity[0])  # laminar flame speed

# ----------------------------------------------------------------------
# plot the fields
plots = [sim.velocity, sim.T, sim.density]
mathnames = ["$u$", "$T$", r"$\rho$", ]
textnames = ["u", "T", "rho", ]
units = ["m/s", "K", "kg/m$^3$", ]

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
plt.legend(fontsize="small")
plt.savefig(save_dir + "pics/field_Xs.png", bbox_inches="tight", dpi=set_dpi)
plt.close()

plt.figure(figsize=(8, 6))
plt.title("Mass Fraction", fontsize="medium")
plt.xlabel("$x$/m")
for i in range(gas.n_species):
    plt.plot(sim.grid, sim.Y[i], lw=2, label=f"{gas.kinetics_species_names[i]}")
plt.legend(fontsize="small")
plt.savefig(save_dir + "pics/field_Ys.png", bbox_inches="tight", dpi=set_dpi)
plt.close()

