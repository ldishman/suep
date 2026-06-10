from interpFromScan_uproot import *
import awkward as ak
import os, sys, glob, re
import ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

###################
# Configure plots #
##################

allPlots = {
    "nPhiGen": Plot(
                    name  = "nPhiGen",
                    var   = lambda x: ak.sum(x["GenPart_pdgId"] == 999999, axis=1),
                    bins  = np.arange(200),
                    extra= {
                        "is2D"   : False,
                        "xlabel" : "N_{#phi}"
                    }
                ),
}

####################
# Helper functions #
####################

def label(path):
    md, T = re.search(r"MD([0-9.]+)_T([0-9.]+)", path).groups()
    return f"m={md}, T={T}"

def get_hist(sample, cfg):
    # same method base.reweight() calls on its otherSample
    if cfg.name not in sample.histos:
        sample.collectHistogram(cfg)
    return sample.histos[cfg.name]

def shape_and_N(h):
    vals = np.array([h.GetBinContent(i+1) for i in range(h.GetNbinsX())])
    return vals, vals.sum()

#####################
# Configure samples #
#####################

variables = ["GenPart_pdgId"]
category = sys.argv[1]
m = float(sys.argv[2])
T = float(sys.argv[3])

# Validated mix (from mix.py) — one sample per point so the variance can be decomposed
mix_folders = [
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_0.25/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T0.25_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_0.35/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T0.35_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_0.50/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T0.50_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_0.71/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T0.71_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T1.00_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_2.83/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T2.83_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_2.0_0.50/UL18/ZHleptonicpythia_leptonic_M125_MD2.0_T0.50_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_2.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD2.0_T1.00_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_1.06/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T1.06_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_4.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD4.0_T1.00_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_6.0_3.00/UL18/ZHleptonicpythia_leptonic_M125_MD6.0_T3.00_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_7.0_7.00/UL18/ZHleptonicpythia_leptonic_M125_MD7.0_T7.00_HT-1_/NANOAOD/",
    "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_8.0_16.00/UL18/ZHleptonicpythia_leptonic_M125_MD8.0_T16.00_HT-1_/NANOAOD/",
]

mix_samples = [
    SUEPSample(
        files = joinLists([collectFilesFromFolder(folder)]),
        tag = label(folder),
        color = ROOT.kRed,
        variables = variables)
    for folder in mix_folders
]

# Held-out validation point from command line (same style as closure.py)
m_str = f"{m:.1f}"
#m_str = f"{m:.2f}"
T_str = f"{T:.2f}"
tmpdir = "/tmp/ldishman/suep_variance"

path = f"/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/{category}_{m_str}_{T_str}/*/*/NANOAOD/"

held_out = SUEPSample(
    files = joinLists([collectFilesFromFolder(folder) for folder in glob.glob(path)]),
    tag = f"Target ({category}) m={m_str}, T={T_str}",
    color = ROOT.kBlue,
    variables = variables)

####################
## Execution area ##
####################

prefix = f"variance_{category}_m{m_str}_T{T_str}_"
outdir = os.path.join(tmpdir, prefix + "output")
os.makedirs(outdir, exist_ok=True)

cfg = allPlots["nPhiGen"]
centers = np.asarray(cfg.bins, dtype=float)[:-1]    # integer-aligned (0.5 input shift)

# Each mix sample's nPhiGen with REAL event counts
sample_counts, sample_labels, sample_N = [], [], []
for s in mix_samples:
    vals, Ns = shape_and_N(get_hist(s, cfg))
    if vals.sum() == 0:
        continue
    sample_counts.append(Ns * vals / vals.sum())    # reconstructed raw counts per bin
    sample_labels.append(s.tag)
    sample_N.append(Ns)
sample_counts = np.array(sample_counts)

print("per-sample event counts (sanity check — should be real Ns):")
for lab, n in zip(sample_labels, sample_N):
    print(f"  {lab:18s} {n:.0f}")

# Actual held-out target as f_target
tvals, _ = shape_and_N(get_hist(held_out, cfg))
f_target = tvals / tvals.sum()

# --- dump raw shapes for the optimizer (one file per target) ---
f_s = sample_counts / sample_counts.sum(axis=1, keepdims=True)   # per-sample N_phi pdfs
np.savez(os.path.join(outdir, prefix + "shapes.npz"),
         f_s=f_s, f_target=f_target,
         labels=np.array(sample_labels), target=f"m={m_str}, T={T_str}")

# Count-weighted mix pdf, per-event weight^2, per-sample variance
mix_counts = sample_counts.sum(axis=0)
f_mix = mix_counts / mix_counts.sum()
w2 = np.divide(f_target, f_mix, out=np.zeros_like(f_target), where=f_mix > 0)**2
contribs = sample_counts * w2                       # n_{s,b} * w(b)^2

# Overlay of mix sample shapes
#for c, lab in zip(sample_counts, sample_labels):
#    plt.step(centers, c / c.sum(), where="mid", label=lab)
#plt.yscale("log")
#plt.xlim(0, 200)
#plt.xlabel(r"$N_\phi$")
#plt.ylabel("Normalized events")
#plt.title("N_φ distributions of mix samples")
#plt.legend(fontsize=7, ncol=2)
#plt.savefig(os.path.join(outdir, prefix + "nphi_overlay.pdf"))
#plt.clf()

# Stacked variance makeup for the target
support = centers[f_target > 1e-3 * f_target.max()]
plt.stackplot(centers, contribs, labels=sample_labels)
plt.xlim(support.min() - 2, support.max() + 2)
plt.xlabel(r"$N_\phi$")
plt.ylabel(r"variance contribution  $n_s\,(f_{target}/f_{mix})^2$")
plt.title(f"Variance makeup, target m={m_str}, T={T_str}")
plt.legend(fontsize=7, ncol=2)
plt.savefig(os.path.join(outdir, prefix + "variance_makeup.pdf"))
plt.clf()

# Real totals: N_eff, relative uncertainty, per-sample shares
V = contribs.sum(axis=1)
Vtot = V.sum()
Nmix = mix_counts.sum()
print(f"\ntarget: rel stat unc = {100*np.sqrt(Vtot)/Nmix:.3f}%   N_eff = {Nmix**2/Vtot:.3e}")
for lab, frac in sorted(zip(sample_labels, V/Vtot), key=lambda z: -z[1]):
    print(f"  {lab:18s} {100*frac:5.1f}% of variance")

gap = (f_mix == 0) & (f_target > 0)
if gap.any():
    print(f"WARNING: {int(gap.sum())} bins have target weight but NO mix events (uncoverable).")
