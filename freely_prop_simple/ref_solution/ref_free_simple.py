"""Reference solution of 1D freely-propagating premixed flames based on simplified physical models."""
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 24
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


# ----------------------------------------------------------------------
# define the constants
W = 28.97e-3  # gas molecular weight, kg/mol
lam = 2.6e-2  # thermal conductivity, W/(m-K)
cp = 1000.0  # heat capacity, J/(kg-K)
qF = 5.0e7  # fuel calorific value, J/kg

R = 8.3145  # universal gas constant, J/(mol-K)
A = 1.4e8  # pre-exponential factor
Ea = 1.214172e5  # activation energy, J/mol
nu_rxn = 1.6  # reaction order

# Rg = 287  # gas constant, J/(kg-K)
Rg = R / W  # gas constant, J/(kg-K)

# ----------------------------------------------------------------------
# set calculation domain
L = 0.0015  # 1.5 mm
n_grids = 10000  # 10, 50, 100, 1000, 10000, 100000, 1000000
n_steps = 500  # maximum iteration steps
dx = L / (n_grids - 1)
x = np.linspace(0, L, n_grids)
# temperature, temperature gradient, velocity, density, pressure, mass fraction of fuel, reaction rate
T, gradT, u, rho, p, YF, omega = np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids), \
                                  np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids)

# ----------------------------------------------------------------------
# inlet boundary condition
T[0] = 298  # K
gradT[0] = 1e5  # K/m
p[0] = 101325 * 1.2  # Pa
# phi = (2 * 32 / 16) * YF[0] / (1 - YF[0])
phi = 0.50
YF[0] = phi / (phi + (2 * 32 / 16))
rho[0] = p[0] / (Rg * T[0])  # kg/m3
omega[0] = A * np.exp(-Ea / (R * T[0])) * (YF[0] * rho[0]) ** nu_rxn  # kg/m3-s

T_max = T[0] + qF * YF[0] / cp

save_dir = "./results/gradT{:.0f}_p{:.2f}_phi{:.4f}/".format(gradT[0], p[0]/101325, phi)
os.makedirs(save_dir + "data/", exist_ok=True)

# ----------------------------------------------------------------------
# solve the problem using the bisection method
u0_l = 0.
u0_r = 1.

t0 = time.perf_counter()
for k_u in range(n_steps):
    print("\nk_u: {:d}, u0_r-u0_l = {:.4e}".format(k_u, u0_r - u0_l))
    # print("omega_max:", max(omega))
    u[0] = (u0_l + u0_r) / 2
    c1 = dx * rho[0] * cp / lam * u[0]
    c2 = dx * qF / lam
    c3 = u[0] + Rg * T[0] / u[0]
    is_converge = True
    for i in range(1, n_grids):
        gradT[i] = gradT[i - 1] + c1 * gradT[i - 1] - c2 * omega[i - 1]
        T[i] = T[i - 1] + dx * gradT[i]
        if gradT[i] < 0:  # flame flashback, indicating a small u0
            u0_l = u[0]
            is_converge = False
            print("flame flashback, u0 too small, i=", i)
            break
        elif T[i] > T_max:  # flame blows out, indicating a large u0
            u0_r = u[0]
            is_converge = False
            print("flame blows out, u0 too large, i=", i)
            break
        else:
            u[i] = 0.5 * (c3 - np.sqrt(c3 ** 2 - 4 * Rg * T[i]))  # choose the smaller root (subsonic)
            rho[i] = rho[0] * u[0] / u[i]
            p[i] = rho[i] * Rg * T[i]
            # p[i] = p[0] - rho[0] * u[0] * (u[i] - u[0])  # same as the last line
            YF[i] = YF[0] + cp * (T[0] - T[i]) / qF
            # YF[i] = cp * (T_max - T[i]) / qF  # same as the last line
            omega[i] = A * np.exp(-Ea / (R * T[i])) * (YF[i] * rho[i]) ** nu_rxn

    if is_converge or u0_r - u0_l < 1e-16:  # the result is sensitive to this criterion
        if i < n_grids - 1:
            print(i)
            T[i:] = T[i - 1]
            gradT[i:] = 0.0
            u[i:], rho[i:], p[i:], YF[i:], omega[i:] = u[i - 1], rho[i - 1], p[i - 1], YF[i - 1], omega[i - 1]
        break

time_cal = time.perf_counter() - t0

print(f"\nsL: {u0_l} m/s")

# ----------------------------------------------------------------------
# plot the fields
fields = [T, YF, u, rho, omega, p - p[0], gradT]
mathnames = ["$T$", "$Y_F$", "$u$", r"$\rho$", r"$\omega$", "$p_{rel}$", r"$\nabla{T}$"]
textnames = ["T", "YF", "u", "rho", "omega", "p", "gradT"]
units = ["K", " ", "m/s", "kg/m$^3$", "kg/(m$^3$·s)", "Pa", "K/m"]

for i in range(len(fields)):
    plt.figure(figsize=(8, 6))
    plt.title(mathnames[i])
    plt.xlabel("$x$/mm")
    plt.ylabel(units[i])
    plt.plot(x * 1e3, fields[i], lw=3)
    plt.savefig(save_dir + f"{i+1}_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)  # .png  .svg
    plt.close()

# ----------------------------------------------------------------------
# save the data
np.save(save_dir + "data/x.npy", x)
np.save(save_dir + "data/T.npy", T)
np.save(save_dir + "data/YF.npy", YF)
np.save(save_dir + "data/u.npy", u)
np.save(save_dir + "data/rho.npy", rho)
np.save(save_dir + "data/omega.npy", omega)
np.save(save_dir + "data/p.npy", p)
np.save(save_dir + "data/gradT.npy", gradT)
np.save(save_dir + "data/sL.npy", u[0])

