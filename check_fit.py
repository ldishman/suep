import uproot
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re, glob

# Gaussian model (identical to fit.py)
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# --- Leptonic-derived fit lines, FIXED (from closure.py) ---
c_fit, a_fit = 100.73, 6.40     # (mu-0.5)*m collapse
A_fit, b_fit = 1.106, 4.129     # sigma/sqrt(mu) collapse

def collapse_model(x, c, a):            # for (mu-0.5)*m
    return c / np.sqrt(1 + a * 2**(2*x))

def sigma_collapse_model(x, A, b):      # for sigma/sqrt(mu)
    return A * 2**x / np.sqrt(1 + b * 2**(2*x))

# Grab every Generic / Hadronic histos.root that exists (NOT Leptonic)
files = sorted(glob.glob("refM3_output/Generic_*_output/*_histos.root") +
               glob.glob("refM3_output/Hadronic_*_output/*_histos.root"))

cats, m_vals, T_vals, mu_vals, sigma_vals = [], [], [], [], []

for fname in files:
    mobj = re.search(r"(Generic|Hadronic)_m([0-9.]+)_T([0-9.]+)", fname)
    if not mobj:
        continue
    cat = mobj.group(1)
    m   = float(mobj.group(2))
    T   = float(mobj.group(3))

    # keep ONLY integer-x grid points x = -2,-1,0,1,2 (drop the sqrt(2) fillers)
    #x = np.log2(T / m)
    #k = round(x)
    #if k not in (-2, -1, 0, 1, 2) or abs(x - k) > 0.1:
    #    continue

    # find the nPhiGen histo by pattern — survives m=1.0 vs m=1.00 naming
    f = uproot.open(fname)
    keys = [key for key in f.keys()
            if key.startswith("nPhiGen") and "_total" in key and "weighted" not in key.lower()]
    if not keys:
        print(f"  no nPhiGen histo in {fname}")
        continue
    h = f[keys[0]]

    # same extraction + 0.5 input-bug shift as fit.py
    values, edges = h.to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:]) - 0.5
    mask = values > 0
    centers, values = centers[mask], values[mask]
    if len(values) < 3:
        continue

    A0 = np.max(values)
    mu0 = centers[np.argmax(values)]
    sigma0 = (centers.max() - centers.min()) / 4
    try:
        popt, _ = curve_fit(gauss, centers, values, p0=[A0, mu0, sigma0])
    except:
        continue
    A, mu, sigma = popt

    cats.append(cat); m_vals.append(m); T_vals.append(T)
    mu_vals.append(mu); sigma_vals.append(sigma)
    print(f"  {cat:8s} m={m:<5} T={T:<6} x={np.log2(T/m):+.2f}  mu={mu:6.2f}  sigma={sigma:6.2f}")

cats   = np.array(cats)
m_vals = np.array(m_vals);   T_vals     = np.array(T_vals)
mu_vals= np.array(mu_vals);  sigma_vals = np.array(sigma_vals)
print(f"{len(m_vals)} points kept")

x_all   = np.log2(T_vals / m_vals)
y_mu    = (mu_vals - 0.5) * m_vals
y_sigma = sigma_vals / np.sqrt(mu_vals)
xs = np.linspace(-2.3, 2.3, 200)
styles = [("Generic", "tab:blue", "o"), ("Hadronic", "tab:red", "^")]

# --- mu plot ---
plt.figure()
for cat, color, marker in styles:
    s = cats == cat
    if s.any():
        plt.scatter(x_all[s], y_mu[s], c=color, marker=marker, s=40, label=cat)
plt.plot(xs, collapse_model(xs, c_fit, a_fit), "k-", label="Leptonic fit")
plt.xlabel("log2(T/m)"); plt.ylabel(r"$(\mu-0.5)\cdot m$")
plt.title("Generic/Hadronic vs Leptonic μ fit"); plt.legend()
plt.savefig("fit_output/validate_mu.png"); plt.savefig("fit_output/validate_mu.pdf")
plt.clf()

# --- sigma plot ---
plt.figure()
for cat, color, marker in styles:
    s = cats == cat
    if s.any():
        plt.scatter(x_all[s], y_sigma[s], c=color, marker=marker, s=40, label=cat)
plt.plot(xs, sigma_collapse_model(xs, A_fit, b_fit), "k-", label="Leptonic fit")
plt.xlabel("log2(T/m)"); plt.ylabel(r"$\sigma/\sqrt{\mu}$")
plt.title("Generic/Hadronic vs Leptonic σ fit"); plt.legend()
plt.savefig("fit_output/validate_sigma.png"); plt.savefig("fit_output/validate_sigma.pdf")
plt.clf()
