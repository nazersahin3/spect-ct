import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def inflation_rhs(tau, u, Wi, Yp0):
    """
    u[0] = Y = phi/phi_i
    u[1] = Y' = dY/dtau
    u[2] = ln(a/a_init)
    """
    Y, Yp, lna = u

    # X = H / H_i from Peacock eq. (11.27)
    denom = Wi + 0.5 * Yp0**2
    X = np.sqrt((Wi * Y**2 + 0.5 * Yp**2) / denom)

    dY_dtau = Yp
    dYp_dtau = -3.0 * X * Yp - 2.0 * Wi * Y   # since dW/dY = 2 Wi Y
    dlna_dtau = X

    return [dY_dtau, dYp_dtau, dlna_dtau]


def solve_model(Wi, tau_max=1000.0, npts=20000):
    # Initial conditions
    Y0 = 1.0
    Yp0 = -(1.0 / 3.0) * (2.0 * Wi * Y0)   # slow-roll initial condition
    lna0 = 0.0

    tau_eval = np.linspace(0.0, tau_max, npts)

    sol = solve_ivp(
        inflation_rhs,
        (0.0, tau_max),
        [Y0, Yp0, lna0],
        args=(Wi, Yp0),
        t_eval=tau_eval,
        rtol=1e-8,
        atol=1e-10,
        method='RK45'
    )

    tau = sol.t
    Y = sol.y[0]
    Yp = sol.y[1]
    lna = sol.y[2]

    # Recompute X and inflation parameter epsilon
    denom = Wi + 0.5 * Yp0**2
    X = np.sqrt((Wi * Y**2 + 0.5 * Yp**2) / denom)

    # For W = Wi Y^2, Peacock states epsilon = eta = (2/3) Wi / Y^2
    epsilon = (2.0 / 3.0) * Wi / (Y**2)

    return tau, Y, Yp, lna, X, epsilon


# Solve for the two cases in the caption
Wi_values = [0.002, 0.005]
results = {}

for Wi in Wi_values:
    results[Wi] = solve_model(Wi)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

# Top panel: phi / phi_init
for Wi in Wi_values:
    tau, Y, Yp, lna, X, epsilon = results[Wi]
    axes[0].plot(tau, np.abs(Y), label=fr"$W_i={Wi}$")

axes[0].set_yscale("log")
axes[0].set_xlim(0, 1000)
axes[0].set_ylim(3e-4, 1)
axes[0].set_ylabel(r"$\phi/\phi_{\rm init}$", fontsize=12)
axes[0].legend()
axes[0].grid(False)

# Bottom panel: ln[a(t)/a_init]
for Wi in Wi_values:
    tau, Y, Yp, lna, X, epsilon = results[Wi]
    axes[1].plot(tau, lna, label=fr"$W_i={Wi}$")

axes[1].set_xlim(0, 1000)
axes[1].set_xlabel(r"$t/H_{\rm init}^{-1}$", fontsize=12)
axes[1].set_ylabel(r"$\ln[a(t)/a_{\rm init}]$", fontsize=12)
axes[1].grid(False)

plt.tight_layout()
plt.show()

# Estimate end of inflation from epsilon = 1
for Wi in Wi_values:
    tau, Y, Yp, lna, X, epsilon = results[Wi]

    idx = np.argmax(epsilon >= 1.0)
    if epsilon[idx] >= 1.0:
        tau_end = tau[idx]
        N_end = lna[idx]
        print(f"Wi = {Wi:.3f}: inflation ends at tau ~ {tau_end:.1f}, N ~ {N_end:.1f}")
    else:
        print(f"Wi = {Wi:.3f}: epsilon never reached 1 by tau = {tau[-1]:.1f}")