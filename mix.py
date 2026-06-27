import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re

# Leptonic
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

# Generic
#mix_files = [
#    "refM3_output/Generic_m2.0_T0.50_output/Generic_m2.0_T0.50_histos.root",  # ~43  (ceiling)
#    "refM3_output/Generic_m2.0_T0.71_output/Generic_m2.0_T0.71_histos.root",  # ~38  (fills 33–42 gap)
#    "refM3_output/Generic_m2.0_T1.00_output/Generic_m2.0_T1.00_histos.root",  # ~32
#    "refM3_output/Generic_m2.0_T1.41_output/Generic_m2.0_T1.41_histos.root",  # ~25
#    "refM3_output/Generic_m2.0_T2.00_output/Generic_m2.0_T2.00_histos.root",  # ~19  (fills 12–19 gap)
#    "refM3_output/Generic_m2.0_T2.83_output/Generic_m2.0_T2.83_histos.root",  # ~14  (fills 12–19 gap)
#    "refM3_output/Generic_m2.0_T4.00_output/Generic_m2.0_T4.00_histos.root",  # ~10
#    "refM3_output/Generic_m2.0_T5.66_output/Generic_m2.0_T5.66_histos.root",  # ~7
#    "refM3_output/Generic_m2.0_T8.00_output/Generic_m2.0_T8.00_histos.root",  # ~5
#    "refM3_output/Generic_m8.0_T16.00_output/Generic_m8.0_T16.00_histos.root",
#    "refM3_output/Generic_m8.0_T32.00_output/Generic_m8.0_T32.00_histos.root", # ~2
#]

# Hadronic
#mix_files = [
#    "refM3_output/Hadronic_m1.40_T0.35_output/Hadronic_m1.40_T0.35_histos.root",  # ~58  (ceiling)
#    "refM3_output/Hadronic_m1.40_T0.49_output/Hadronic_m1.40_T0.49_histos.root",  # ~51
#    "refM3_output/Hadronic_m1.40_T0.70_output/Hadronic_m1.40_T0.70_histos.root",  # ~43
#    "refM3_output/Hadronic_m1.40_T0.99_output/Hadronic_m1.40_T0.99_histos.root",  # ~35
#    "refM3_output/Hadronic_m1.40_T1.40_output/Hadronic_m1.40_T1.40_histos.root",  # ~27
#    "refM3_output/Hadronic_m1.40_T1.98_output/Hadronic_m1.40_T1.98_histos.root",  # ~20
#    "refM3_output/Hadronic_m1.40_T2.80_output/Hadronic_m1.40_T2.80_histos.root",  # ~14
#    "refM3_output/Hadronic_m1.40_T3.96_output/Hadronic_m1.40_T3.96_histos.root",  # ~10
#    "refM3_output/Hadronic_m1.40_T5.60_output/Hadronic_m1.40_T5.60_histos.root",  # ~7
#    "refM3_output/Hadronic_m8.0_T16.00_output/Hadronic_m8.0_T16.00_histos.root",  # ~3
#    "refM3_output/Hadronic_m8.0_T32.00_output/Hadronic_m8.0_T32.00_histos.root",  # ~2
#]

combined = None
centers = None

for fname in mix_files:
    m = float(re.search(r"m([0-9.]+)", fname).group(1))
    T = float(re.search(r"T([0-9.]+)", fname).group(1))

    file = uproot.open(fname)
    #hname = f"nPhiGen_m={m}, T={T:.2f}_total"
    #if hname not in file:
    #    continue

    #values, edges = file[hname].to_numpy()
    keys = [k for k in file.keys()
            if k.startswith("nPhiGen") and "_total" in k and "weighted" not in k.lower()]
    if not keys:
        continue
    values, edges = file[keys[0]].to_numpy()
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
