import uproot
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re

# Gaussian model
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Input files 
files = [
    "refM3_output/Leptonic_m1.0_T0.25_output/Leptonic_m1.0_T0.25_histos.root",
    "refM3_output/Leptonic_m1.0_T0.50_output/Leptonic_m1.0_T0.50_histos.root",
    "refM3_output/Leptonic_m1.0_T1.00_output/Leptonic_m1.0_T1.00_histos.root",
    "refM3_output/Leptonic_m1.0_T2.00_output/Leptonic_m1.0_T2.00_histos.root",
    "refM3_output/Leptonic_m1.0_T4.00_output/Leptonic_m1.0_T4.00_histos.root",

    "refM3_output/Leptonic_m2.0_T0.50_output/Leptonic_m2.0_T0.50_histos.root",
    "refM3_output/Leptonic_m2.0_T1.00_output/Leptonic_m2.0_T1.00_histos.root",
    "refM3_output/Leptonic_m2.0_T2.00_output/Leptonic_m2.0_T2.00_histos.root",
    "refM3_output/Leptonic_m2.0_T4.00_output/Leptonic_m2.0_T4.00_histos.root",
    "refM3_output/Leptonic_m2.0_T8.00_output/Leptonic_m2.0_T8.00_histos.root",

    "refM3_output/Leptonic_m3.0_T0.75_output/Leptonic_m3.0_T0.75_histos.root",
    "refM3_output/Leptonic_m3.0_T1.50_output/Leptonic_m3.0_T1.50_histos.root",
    "refM3_output/Leptonic_m3.0_T3.00_output/Leptonic_m3.0_T3.00_histos.root",
    "refM3_output/Leptonic_m3.0_T6.00_output/Leptonic_m3.0_T6.00_histos.root",
    "refM3_output/Leptonic_m3.0_T12.00_output/Leptonic_m3.0_T12.00_histos.root",

    "refM3_output/Leptonic_m4.0_T1.00_output/Leptonic_m4.0_T1.00_histos.root",
    "refM3_output/Leptonic_m4.0_T2.00_output/Leptonic_m4.0_T2.00_histos.root",
    "refM3_output/Leptonic_m4.0_T4.00_output/Leptonic_m4.0_T4.00_histos.root",
    "refM3_output/Leptonic_m4.0_T8.00_output/Leptonic_m4.0_T8.00_histos.root",
    "refM3_output/Leptonic_m4.0_T16.00_output/Leptonic_m4.0_T16.00_histos.root",

    "refM3_output/Leptonic_m5.0_T1.25_output/Leptonic_m5.0_T1.25_histos.root",
    "refM3_output/Leptonic_m5.0_T2.50_output/Leptonic_m5.0_T2.50_histos.root",
    "refM3_output/Leptonic_m5.0_T5.00_output/Leptonic_m5.0_T5.00_histos.root",
    "refM3_output/Leptonic_m5.0_T10.00_output/Leptonic_m5.0_T10.00_histos.root",
    "refM3_output/Leptonic_m5.0_T20.00_output/Leptonic_m5.0_T20.00_histos.root",

    "refM3_output/Leptonic_m6.0_T1.50_output/Leptonic_m6.0_T1.50_histos.root",
    "refM3_output/Leptonic_m6.0_T3.00_output/Leptonic_m6.0_T3.00_histos.root",
    "refM3_output/Leptonic_m6.0_T6.00_output/Leptonic_m6.0_T6.00_histos.root",
    "refM3_output/Leptonic_m6.0_T12.00_output/Leptonic_m6.0_T12.00_histos.root",
    "refM3_output/Leptonic_m6.0_T24.00_output/Leptonic_m6.0_T24.00_histos.root",

    "refM3_output/Leptonic_m7.0_T1.75_output/Leptonic_m7.0_T1.75_histos.root",
    "refM3_output/Leptonic_m7.0_T3.50_output/Leptonic_m7.0_T3.50_histos.root",
    "refM3_output/Leptonic_m7.0_T7.00_output/Leptonic_m7.0_T7.00_histos.root",
    "refM3_output/Leptonic_m7.0_T14.00_output/Leptonic_m7.0_T14.00_histos.root",
    "refM3_output/Leptonic_m7.0_T28.00_output/Leptonic_m7.0_T28.00_histos.root",

    "refM3_output/Leptonic_m8.0_T2.00_output/Leptonic_m8.0_T2.00_histos.root",
    "refM3_output/Leptonic_m8.0_T4.00_output/Leptonic_m8.0_T4.00_histos.root",
    "refM3_output/Leptonic_m8.0_T8.00_output/Leptonic_m8.0_T8.00_histos.root",
    "refM3_output/Leptonic_m8.0_T16.00_output/Leptonic_m8.0_T16.00_histos.root",
    "refM3_output/Leptonic_m8.0_T32.00_output/Leptonic_m8.0_T32.00_histos.root"
]

# Storage for results
m_vals, T_vals = [], []
A_vals, mu_vals, sigma_vals = [], [], []

# Set up summary figure
fig_all, axes_all = plt.subplots(8, 5, figsize=(15, 20))
plt.figure()

# Loop over files
for fname in files:

    # Extract m and T from filename
    m_match = re.search(r"m([0-9.]+)", fname)
    T_match = re.search(r"T([0-9.]+)", fname)

    if not (m_match and T_match):
        continue

    m = float(m_match.group(1))
    T = float(T_match.group(1))

    # Open file
    file = uproot.open(fname)

    # Build histogram name dynamically
    hname = f"nPhiGen_m={m}, T={T:.2f}_total"

    if hname not in file:
        continue

    h = file[hname]

    # Extract bin data
    values, edges = h.to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers = centers - 0.5      # shift all plots left by 0.5 -- input bug

    # Remove zero bins
    mask = values > 0
    centers = centers[mask]
    values = values[mask]

    if len(values) < 3:
        continue

    # Initial guesses
    A0 = np.max(values)
    mu0 = centers[np.argmax(values)]
    sigma0 = (centers.max() - centers.min()) / 4

    # Fit
    try:
        popt, _ = curve_fit(gauss, centers, values, p0=[A0, mu0, sigma0])
    except:
        continue

    A, mu, sigma = popt

    # Goodness of fit (R²)
    residuals = values - gauss(centers, A, mu, sigma)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((values - np.mean(values))**2)
    r2 = 1 - ss_res / ss_tot

    # Store results
    m_vals.append(m)
    T_vals.append(T)
    A_vals.append(A)
    mu_vals.append(mu)
    sigma_vals.append(sigma)

    # Make per-sample plot
    x_fit = np.linspace(min(centers), max(centers), 500)
    y_fit = gauss(x_fit, A, mu, sigma)
    plt.scatter(centers, values, color="blue", label="Histogram")
    plt.plot(x_fit, y_fit, color="red", label="Gaussian fit")
    # Add parameters to individual plot
    plt.text(0.05, 0.95,
             f"A = {A:.2f}\nμ = {mu:.2f}\nσ = {sigma:.2f}\nR² = {r2:.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top')

    plt.xlabel("nPhiGen")
    plt.ylabel("Normalized Events")
    plt.title(f"m={m}, T={T}")
    plt.legend()

    # Add individual plot to summary figure 
    ax = axes_all.flatten()[len(mu_vals)-1]
    ax.scatter(centers, values, s=5, color="blue", label="Histogram")
    ax.plot(x_fit, y_fit, color="red", label="Gaussian fit")
    ax.text(0.05, 0.95,
             f"A = {A:.2f}\nμ = {mu:.2f}\nσ = {sigma:.2f}\nR² = {r2:.3f}",
             transform=ax.transAxes,
             verticalalignment='top', fontsize=6)
    ax.set_xlabel("nPhiGen")
    ax.set_ylabel("Normalized Events")
    ax.set_title(f"m={m}, T={T}")
    ax.legend()

    # Save individual plot
    outname = f"fit_output/fit_m{m:.1f}_T{T:.2f}.png"
    plt.savefig(outname)  # png
    plt.savefig(outname.replace(".png", ".pdf"))  # pdf
    plt.clf()

# Save summary figure
fig_all.tight_layout()
fig_all.savefig("fit_output/all_fits.png")
fig_all.savefig("fit_output/all_fits.pdf")

# Convert to arrays
m_vals = np.array(m_vals)
T_vals = np.array(T_vals)
mu_vals = np.array(mu_vals)
sigma_vals = np.array(sigma_vals)

##################################### End section on fitting sample histos
##################################### Begin section on plotting, fitting μ, σ

# Universal fit: μ·m = c / sqrt(1 + a*2^(2x))
def collapse_model(x, c, a):
    return c / np.sqrt(1 + a * 2**(2*x))

x_all = np.log2(T_vals / m_vals)
y_all = mu_vals * m_vals

popt, _ = curve_fit(collapse_model, x_all, y_all, p0=[90.0, 1.0])
c_fit, a_fit = popt
y_pred = collapse_model(x_all, *popt)
ss_res = np.sum((y_all - y_pred)**2)
ss_tot = np.sum((y_all - np.mean(y_all))**2)
r2 = 1 - ss_res / ss_tot
print(f"μ·m fit: c = {c_fit:.2f}, a = {a_fit:.2f}, R² = {r2:.4f}")

x_smooth = np.linspace(x_all.min(), x_all.max(), 200)
plt.scatter(x_all, y_all, c=m_vals, label="data")
plt.colorbar(label="m")
plt.plot(x_smooth, collapse_model(x_smooth, *popt), "r-", label=f"fit: c={c_fit:.2f}, a={a_fit:.2f}, R²={r2:.3f}")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$\mu \cdot m$")
plt.title(f"Fit: μ·m = c / √(1 + a·2^(2x))")
plt.legend()
plt.savefig("fit_output/mu_universal_fit.png")
plt.savefig("fit_output/mu_universal_fit.pdf")
plt.clf()

# Same fit, but with a (μ−0.5) phenomenological offset, exclude x=2
mask = x_all < 1.9
y_all = (mu_vals - 0.5) * m_vals

popt, _ = curve_fit(collapse_model, x_all[mask], y_all[mask], p0=[90.0, 1.0])
c_fit, a_fit = popt
y_pred = collapse_model(x_all[mask], *popt)
ss_res = np.sum((y_all[mask] - y_pred)**2)
ss_tot = np.sum((y_all[mask] - np.mean(y_all[mask]))**2)
r2 = 1 - ss_res / ss_tot
print(f"(μ−0.5)·m fit: c = {c_fit:.2f}, a = {a_fit:.2f}, R² = {r2:.4f}")

plt.scatter(x_all, y_all, c=m_vals, label="data")
plt.colorbar(label="m")
plt.plot(x_smooth, collapse_model(x_smooth, *popt), "r-", label=f"fit: c={c_fit:.2f}, a={a_fit:.2f}, R²={r2:.3f}")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$(\mu-0.5)\cdot m$")
plt.title("Fit: (μ−0.5)·m = c / √(1 + a·2^(2x)) (x=2 excluded)")
plt.legend()
plt.savefig("fit_output/mu_offset_universal_fit.png")
plt.savefig("fit_output/mu_offset_universal_fit.pdf")
plt.clf()


# Universal σ/√μ fit, exclude x=2 column where Gaussian-fit breaks down
ratio_all = sigma_vals / np.sqrt(mu_vals)

def sigma_collapse_model(x, A, b):
    return A * 2**x / np.sqrt(1 + b * 2**(2*x))

popt_s, _ = curve_fit(sigma_collapse_model, x_all[mask], ratio_all[mask], p0=[1.0, 3.0])
A_fit, b_fit = popt_s

y_pred_s = sigma_collapse_model(x_all[mask], *popt_s)
r2_s = 1 - np.sum((ratio_all[mask] - y_pred_s)**2) / np.sum((ratio_all[mask] - np.mean(ratio_all[mask]))**2)
print(f"σ/√μ fit: A = {A_fit:.4f}, b = {b_fit:.4f}, R² = {r2_s:.4f}")

plt.scatter(x_all, ratio_all, c=m_vals, label="data")
plt.colorbar(label="m")
plt.plot(x_smooth, sigma_collapse_model(x_smooth, *popt_s), "r-",
         label=f"fit: A={A_fit:.3f}, b={b_fit:.3f}, R²={r2_s:.3f}")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$\sigma/\sqrt{\mu}$")
plt.title("Fit: σ/√μ = A · 2^x / √(1 + b · 2^(2x)) (x=2 excluded)")
plt.legend()
plt.savefig("fit_output/sigma_collapse_fit.png")
plt.savefig("fit_output/sigma_collapse_fit.pdf")
plt.clf()
