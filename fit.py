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

def mu_model(T, a, c):
    #return c / np.sqrt(4.0**2 + a * T**2)   # m = 4 fixed
    return a*(1.0/T) + c

def sigma_quad(T, a, b, c):
    return a*T**2 + b*T + c

def sigma_model(T, A, T0, w, c):
    return A * np.exp(-(T - T0)**2 / (2*w**2)) + c

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

    # Add parameters to plot
    plt.text(0.05, 0.95,
             f"A = {A:.2f}\nμ = {mu:.2f}\nσ = {sigma:.2f}\nR² = {r2:.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top')

    plt.xlabel("nPhiGen")
    plt.ylabel("Normalized Events")
    plt.title(f"m={m}, T={T}")
    plt.legend()

    # Save plot
    outname = f"fit_output/fit_m{m:.1f}_T{T:.2f}.png"
    plt.savefig(outname)  # png
    plt.savefig(outname.replace(".png", ".pdf"))  # pdf
    plt.clf()

# Convert to arrays
m_vals = np.array(m_vals)
T_vals = np.array(T_vals)
mu_vals = np.array(mu_vals)
sigma_vals = np.array(sigma_vals)

# μ and σ vs log2(T/m) — all samples
log2_ratio = np.log2(T_vals / m_vals)

plt.scatter(log2_ratio, mu_vals, c=m_vals)
plt.colorbar(label="m")
plt.xlabel("log2(T/m)")
plt.ylabel("Mean μ")
plt.title("μ vs log2(T/m) (all samples)")
plt.savefig("fit_output/mu_vs_log2T_over_m_all.png")
plt.savefig("fit_output/mu_vs_log2T_over_m_all.pdf")
plt.clf()

plt.scatter(log2_ratio, sigma_vals, c=m_vals)
plt.colorbar(label="m")
plt.xlabel("log2(T/m)")
plt.ylabel("Sigma σ")
plt.title("σ vs log2(T/m) (all samples)")
plt.savefig("fit_output/sigma_vs_log2T_over_m_all.png")
plt.savefig("fit_output/sigma_vs_log2T_over_m_all.pdf")
plt.clf()

# μ vs T (fixed m ≈ 1)
mask = np.isclose(m_vals, 1.0)

plt.scatter(T_vals[mask], mu_vals[mask])
plt.xlabel("Temperature T")
plt.ylabel("Mean μ")
plt.title("μ vs T (m=1)")
plt.savefig("fit_output/mu_vs_T.png")
plt.savefig("fit_output/mu_vs_T.pdf")
plt.clf()

# σ vs T (fixed m ≈ 1) 
plt.scatter(T_vals[mask], sigma_vals[mask])
plt.xlabel("Temperature T")
plt.ylabel("Sigma σ")
plt.title("σ vs T (m=1)")
plt.savefig("fit_output/sigma_vs_T.png")
plt.savefig("fit_output/sigma_vs_T.pdf")
plt.clf()

# μ vs m (fixed T ≈ 4)
mask = np.isclose(T_vals, 4.0, atol=0.51)

plt.scatter(m_vals[mask], mu_vals[mask])
plt.xlabel("Mass m")
plt.ylabel("Mean μ")
plt.title("μ vs m (T=4)")
plt.savefig("fit_output/mu_vs_m.png")
plt.savefig("fit_output/mu_vs_m.pdf")
plt.clf()

# σ vs m (fixed T ≈ 4)
plt.scatter(m_vals[mask], sigma_vals[mask])
plt.xlabel("Mass m")
plt.ylabel("Sigma σ")
plt.title("σ vs m (T=4)")
plt.savefig("fit_output/sigma_vs_m.png")
plt.savefig("fit_output/sigma_vs_m.pdf")
plt.clf()

# Filter for T/m fit -- m=4 for now
mask = np.isclose(m_vals, 4.0)
T_m4 = T_vals[mask]
mu_m4 = mu_vals[mask]
sigma_m4 = sigma_vals[mask]
log2_T_over_m = np.log2(T_m4 / 4.0)

# μ vs log2(T/m) -- m=4
plt.scatter(log2_T_over_m, mu_m4, label="data")
#popt, _ = curve_fit(mu_model, T_m4, mu_m4, p0=[1.0, 1.0], bounds=([0, 0], [np.inf, np.inf]))  # a ≥ 0, c ≥ 0)
popt, _ = curve_fit(mu_model, T_m4, mu_m4, p0=[1.0, 1.0])  # a ≥ 0, c ≥ 0)
T_fit = np.linspace(min(T_m4), max(T_m4), 200)
log2_fit = np.log2(T_fit / 4.0)
plt.plot(log2_fit, mu_model(T_fit, *popt), color="red", label="fit")
plt.xlabel("log2(T/m)")
plt.ylabel("Mean μ")
plt.title("μ vs log2(T/m) (m=4)")
plt.legend()
plt.savefig("fit_output/mu_vs_log2T_over_m_m4.png")
plt.savefig("fit_output/mu_vs_log2T_over_m_m4.pdf")
plt.clf()

# σ vs log2(T/m) -- m=4
#plt.scatter(log2_T_over_m, sigma_m4, label="data")
#popt, _ = curve_fit(sigma_quad, T_m4, sigma_m4)
popt_gauss, _ = curve_fit(sigma_model, T_m4, sigma_m4, p0=[1.0, np.mean(T_m4), 1.0, np.min(sigma_m4)])
T_fit = np.linspace(min(T_m4), max(T_m4), 200)
#log2_fit = np.log2(T_fit / 4.0)
#plt.plot(log2_fit, sigma_quad(T_fit, *popt), color="red", label="quadratic fit")
#plt.xlabel("log2(T/m)")
#plt.ylabel("Sigma σ")
#plt.title("σ vs log2(T/m) (m=4)")
#plt.legend()
#plt.savefig("fit_output/sigma_vs_log2T_over_m_m4.png")
#plt.savefig("fit_output/sigma_vs_log2T_over_m_m4.pdf")
#plt.clf()

plt.scatter(T_m4, sigma_m4, label="data")
plt.plot(T_fit, sigma_model(T_fit, *popt_gauss), color="red", label="Gaussian fit")

plt.xlabel("T")
plt.ylabel("σ")
plt.title("σ vs T (Gaussian fit)")
plt.legend()

plt.savefig("fit_output/sigma_fit_gauss.png")
plt.savefig("fit_output/sigma_fit_gauss.pdf")
plt.clf()

plt.scatter(log2_T_over_m, sigma_m4, label="data")

popt_sigma, _ = curve_fit(
    sigma_model,
    T_m4,
    sigma_m4,
    p0=[1.0, np.mean(T_m4), 1.0, np.min(sigma_m4)]
)

#T_fit = np.linspace(min(T_m4), max(T_m4), 200)
log2_fit = np.log2(T_fit / 4.0)

plt.plot(log2_fit,
         sigma_model(T_fit, *popt_sigma),
         color="red",
         label="Gaussian fit")

plt.xlabel("log2(T/m)")
plt.ylabel("Sigma σ")
plt.title("σ vs log2(T/m) (m=4)")
plt.legend()

plt.savefig("fit_output/sigma_vs_log2T_over_m_m4.png")
plt.savefig("fit_output/sigma_vs_log2T_over_m_m4.pdf")
plt.clf()

# σ vs μ (colored by log2(T/m)) -- m=4
plt.scatter(mu_m4, sigma_m4, c=log2_T_over_m)
plt.colorbar(label="log2(T/m)")
plt.xlabel("μ")
plt.ylabel("σ")
plt.title("σ vs μ (m=4 only)")
plt.savefig("fit_output/sigma_vs_mu_m4.png")
plt.savefig("fit_output/sigma_vs_mu_m4.pdf")
plt.clf()

from scipy.interpolate import RBFInterpolator

x_vals = np.log2(T_vals / m_vals)
points = np.column_stack([x_vals, m_vals])
mu_interp    = RBFInterpolator(points, mu_vals,    smoothing=0.0)
sigma_interp = RBFInterpolator(points, sigma_vals, smoothing=0.0)

def predict_gauss(T, m):
    """Return (μ, σ) at any (T, m)."""
    p = np.array([[np.log2(T/m), m]])
    return float(mu_interp(p)[0]), float(sigma_interp(p)[0])

# Sanity check: predict at training points, see if it matches
mu_pred = mu_interp(points)
print("μ interp residual max:", np.max(np.abs(mu_pred - mu_vals)))

# Visualize the interpolation surfaces
x_grid = np.linspace(x_vals.min(), x_vals.max(), 100)
m_grid = np.linspace(m_vals.min(), m_vals.max(), 100)
XX, MM = np.meshgrid(x_grid, m_grid)
grid_points = np.column_stack([XX.ravel(), MM.ravel()])

mu_surface    = mu_interp(grid_points).reshape(XX.shape)
sigma_surface = sigma_interp(grid_points).reshape(XX.shape)

plt.pcolormesh(XX, MM, mu_surface, shading="auto")
plt.colorbar(label="μ")
plt.scatter(x_vals, m_vals, c=mu_vals, edgecolor="white", linewidth=0.7)
plt.xlabel("log2(T/m)"); plt.ylabel("m"); plt.title("μ(T, m) interpolation")
plt.savefig("fit_output/mu_surface.png"); plt.savefig("fit_output/mu_surface.pdf"); plt.clf()

plt.pcolormesh(XX, MM, sigma_surface, shading="auto")
plt.colorbar(label="σ")
plt.scatter(x_vals, m_vals, c=sigma_vals, edgecolor="white", linewidth=0.7)
plt.xlabel("log2(T/m)"); plt.ylabel("m"); plt.title("σ(T, m) interpolation")
plt.savefig("fit_output/sigma_surface.png"); plt.savefig("fit_output/sigma_surface.pdf"); plt.clf()

# Overlay nPhi histograms from selected samples
#overlay_files = [
#    "refM3_output/Leptonic_m1.0_T1.00_output/Leptonic_m1.0_T1.00_histos.root",
#    #"refM3_output/Leptonic_m2.0_T5.66_output/Leptonic_m2.0_T5.66_histos.root",
#    "refM3_output/Leptonic_m2.0_T1.00_output/Leptonic_m2.0_T1.00_histos.root",
#    "refM3_output/Leptonic_m3.0_T1.06_output/Leptonic_m3.0_T1.06_histos.root",
#    "refM3_output/Leptonic_m3.0_T4.24_output/Leptonic_m3.0_T4.24_histos.root",
#    "refM3_output/Leptonic_m3.0_T8.49_output/Leptonic_m3.0_T8.49_histos.root",
#    "refM3_output/Leptonic_m3.0_T12.00_output/Leptonic_m3.0_T12.00_histos.root",
#    "refM3_output/Leptonic_m4.0_T16.00_output/Leptonic_m4.0_T16.00_histos.root",
#    "refM3_output/Leptonic_m5.0_T20.00_output/Leptonic_m5.0_T20.00_histos.root",
#    "refM3_output/Leptonic_m6.0_T24.00_output/Leptonic_m6.0_T24.00_histos.root",
#    "refM3_output/Leptonic_m6.0_T3.00_output/Leptonic_m6.0_T3.00_histos.root",
#    "refM3_output/Leptonic_m7.0_T28.00_output/Leptonic_m7.0_T28.00_histos.root",
#    #"refM3_output/Leptonic_m7.0_T19.80_output/Leptonic_m7.0_T19.80_histos.root",
#    "refM3_output/Leptonic_m8.0_T8.00_output/Leptonic_m8.0_T8.00_histos.root",
#    "refM3_output/Leptonic_m8.0_T16.00_output/Leptonic_m8.0_T16.00_histos.root",
#    "refM3_output/Leptonic_m8.0_T32.00_output/Leptonic_m8.0_T32.00_histos.root",
#]
#
#for fname in overlay_files:
#    m_match = re.search(r"m([0-9.]+)", fname)
#    T_match = re.search(r"T([0-9.]+)", fname)
#    if not (m_match and T_match):
#        continue
#    m = float(m_match.group(1))
#    T = float(T_match.group(1))
#
#    file = uproot.open(fname)
#    hname = f"nPhiGen_m={m}, T={T:.2f}_total"
#    if hname not in file:
#        continue
#
#    values, edges = file[hname].to_numpy()
#    centers = 0.5 * (edges[:-1] + edges[1:])
#    plt.step(centers, values, where="mid", label=f"m={m}, T={T:.2f}")
#
#plt.yscale("log")
#plt.xlabel(r"$N_\phi$")
#plt.ylabel("Normalized events")
#plt.legend(fontsize=8)
#plt.savefig("fit_output/nphi_overlay.png")
#plt.savefig("fit_output/nphi_overlay.pdf")
#plt.clf()

# Collapse test: μ·m vs log2(T/m) should fall on one universal curve if μ ∝ 1/m
plt.scatter(np.log2(T_vals / m_vals), mu_vals * m_vals, c=m_vals)
plt.colorbar(label="m")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$\mu \cdot m$")
plt.title("Collapse test: μ·m vs log2(T/m)")
plt.savefig("fit_output/mu_times_m_collapse.png")
plt.savefig("fit_output/mu_times_m_collapse.pdf")
plt.clf()

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
print(f"R² = {r2:.4f}")
print(f"c = {c_fit:.2f}, a = {a_fit:.2f}")

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

# σ/μ collapse + linear fit
ratio_all = sigma_vals / mu_vals

def ratio_model(x, p, q):
    return p*x + q

popt_r, _ = curve_fit(ratio_model, x_all, ratio_all, p0=[0.05, 0.15])
p_fit, q_fit = popt_r

y_pred_r = ratio_model(x_all, *popt_r)
r2_r = 1 - np.sum((ratio_all - y_pred_r)**2) / np.sum((ratio_all - np.mean(ratio_all))**2)
print(f"σ/μ fit: p = {p_fit:.4f}, q = {q_fit:.4f}, R² = {r2_r:.4f}")

plt.scatter(x_all, ratio_all, c=m_vals, label="data")
plt.colorbar(label="m")
plt.plot(x_smooth, ratio_model(x_smooth, *popt_r), "r-",
         label=f"fit: p={p_fit:.3f}, q={q_fit:.3f}, R²={r2_r:.3f}")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$\sigma/\mu$")
plt.title("σ/μ linear fit")
plt.legend()
plt.savefig("fit_output/sigma_over_mu_fit.png")
plt.savefig("fit_output/sigma_over_mu_fit.pdf")
plt.clf()

ratio2_all = sigma_vals / np.sqrt(mu_vals)

plt.scatter(x_all, ratio2_all, c=m_vals)
plt.colorbar(label="m")
plt.xlabel("log2(T/m)")
plt.ylabel(r"$\sigma/\sqrt{\mu}$")
plt.title("σ/√μ collapse test")
plt.savefig("fit_output/sigma_over_sqrtmu.png")
plt.savefig("fit_output/sigma_over_sqrtmu.pdf")
plt.clf()

# Exclude x=2 column where Gaussian-fit breaks down
mask = x_all < 1.9

def sigma_collapse_model(x, A, b):
    return A * 2**x / np.sqrt(1 + b * 2**(2*x))

popt_s, _ = curve_fit(sigma_collapse_model, x_all[mask], ratio2_all[mask], p0=[1.0, 3.0])
A_fit, b_fit = popt_s

y_pred_s = sigma_collapse_model(x_all[mask], *popt_s)
r2_s = 1 - np.sum((ratio2_all[mask] - y_pred_s)**2) / np.sum((ratio2_all[mask] - np.mean(ratio2_all[mask]))**2)
print(f"σ/√μ fit: A = {A_fit:.4f}, b = {b_fit:.4f}, R² = {r2_s:.4f}")

plt.scatter(x_all, ratio2_all, c=m_vals, label="data")
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
