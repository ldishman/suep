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
    "refM3_output/Leptonic_m1.0_T1.00_output/Leptonic_m1.0_T1.00_histos.root",
    "refM3_output/Leptonic_m1.0_T1.41_output/Leptonic_m1.0_T1.41_histos.root",
    "refM3_output/Leptonic_m1.0_T2.00_output/Leptonic_m1.0_T2.00_histos.root",
    "refM3_output/Leptonic_m1.0_T2.83_output/Leptonic_m1.0_T2.83_histos.root",
    "refM3_output/Leptonic_m1.0_T4.00_output/Leptonic_m1.0_T4.00_histos.root",
    "refM3_output/Leptonic_m2.0_T4.00_output/Leptonic_m2.0_T4.00_histos.root",
    "refM3_output/Leptonic_m3.0_T4.24_output/Leptonic_m3.0_T4.24_histos.root",
    "refM3_output/Leptonic_m4.0_T4.00_output/Leptonic_m4.0_T4.00_histos.root",
    "refM3_output/Leptonic_m5.0_T3.54_output/Leptonic_m5.0_T3.54_histos.root",
    "refM3_output/Leptonic_m6.0_T4.24_output/Leptonic_m6.0_T4.24_histos.root",
    "refM3_output/Leptonic_m7.0_T3.50_output/Leptonic_m7.0_T3.50_histos.root",
    "refM3_output/Leptonic_m8.0_T4.00_output/Leptonic_m8.0_T4.00_histos.root",
    "refM3_output/Leptonic_m4.0_T1.00_output/Leptonic_m4.0_T1.00_histos.root",
    "refM3_output/Leptonic_m4.0_T11.31_output/Leptonic_m4.0_T11.31_histos.root",
    "refM3_output/Leptonic_m4.0_T1.41_output/Leptonic_m4.0_T1.41_histos.root",
    "refM3_output/Leptonic_m4.0_T16.00_output/Leptonic_m4.0_T16.00_histos.root",
    "refM3_output/Leptonic_m4.0_T2.00_output/Leptonic_m4.0_T2.00_histos.root",
    "refM3_output/Leptonic_m4.0_T2.83_output/Leptonic_m4.0_T2.83_histos.root",
    "refM3_output/Leptonic_m4.0_T5.66_output/Leptonic_m4.0_T5.66_histos.root",
    "refM3_output/Leptonic_m4.0_T8.00_output/Leptonic_m4.0_T8.00_histos.root"
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
             f"A = {A:.2f}\nμ = {mu:.2f}\nσ = {sigma:.2f}",
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
