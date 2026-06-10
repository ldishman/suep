import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re

mix_files = [
    "refM3_output/Leptonic_m1.0_T0.25_output/Leptonic_m1.0_T0.25_histos.root",
    "refM3_output/Leptonic_m1.0_T0.35_output/Leptonic_m1.0_T0.35_histos.root",
    "refM3_output/Leptonic_m1.0_T0.50_output/Leptonic_m1.0_T0.50_histos.root",
    "refM3_output/Leptonic_m1.0_T0.71_output/Leptonic_m1.0_T0.71_histos.root",
    "refM3_output/Leptonic_m1.0_T1.00_output/Leptonic_m1.0_T1.00_histos.root",
    "refM3_output/Leptonic_m1.0_T2.83_output/Leptonic_m1.0_T2.83_histos.root",  # fills ~15-18 gap
    "refM3_output/Leptonic_m2.0_T0.50_output/Leptonic_m2.0_T0.50_histos.root",
    "refM3_output/Leptonic_m2.0_T1.00_output/Leptonic_m2.0_T1.00_histos.root",
    "refM3_output/Leptonic_m3.0_T1.06_output/Leptonic_m3.0_T1.06_histos.root",
    "refM3_output/Leptonic_m4.0_T1.00_output/Leptonic_m4.0_T1.00_histos.root",
    "refM3_output/Leptonic_m6.0_T3.00_output/Leptonic_m6.0_T3.00_histos.root",
    "refM3_output/Leptonic_m7.0_T7.00_output/Leptonic_m7.0_T7.00_histos.root",
    "refM3_output/Leptonic_m8.0_T16.00_output/Leptonic_m8.0_T16.00_histos.root",
]

combined = None
centers = None

for fname in mix_files:
    m = float(re.search(r"m([0-9.]+)", fname).group(1))
    T = float(re.search(r"T([0-9.]+)", fname).group(1))

    file = uproot.open(fname)
    hname = f"nPhiGen_m={m}, T={T:.2f}_total"
    if hname not in file:
        continue

    values, edges = file[hname].to_numpy()
    #centers = 0.5 * (edges[:-1] + edges[1:])
    centers = edges[:-1]    # numpy.ndarray
    plt.step(centers, values, where="mid", label=f"m={m}, T={T:.2f}")    # i.e. friendlier histogram

    if combined is None:
        combined = np.zeros_like(values)
    combined += values

plt.yscale("log")
plt.xlim(0, 200)
plt.xlabel(r"$N_\phi$")
plt.ylabel("Normalized events")
plt.title("N_φ distributions of mix samples")
plt.legend(fontsize=8)
plt.savefig("fit_output/nphi_overlay.pdf")
plt.clf()

cdf = np.cumsum(combined) / combined.sum()

plt.step(centers, cdf, where="mid")
plt.xlim(0, 200)
plt.xlabel(r"$N_\phi$")
plt.ylabel("Cumulative fraction")
plt.title("CDF of combined mix")
plt.savefig("fit_output/nphi_cdf.pdf")
plt.clf()
