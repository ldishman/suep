from interpFromScan_uproot import *
import awkward as ak
import os, sys, glob
from functools import partial
import ROOT
import numpy as np
import array
import psutil, os

###################
# Configure plots #
##################

def GenSphericity(events):
    ############# Begin lab frame sphericity
    #cut = (events["GenPart_pdgId"]==999999)
    ## Only the dark mesons
    #pt   = events["GenPart_pt"][cut]
    #eta  = events["GenPart_eta"][cut]
    #phi  = events["GenPart_phi"][cut]
    ## Convert to euclidean
    #px = pt*np.cos(phi)
    #py = pt*np.sin(phi)
    #pz = pt*np.sinh(eta)
    #p  = pt*np.cosh(eta)

    ## Sphericity magic
    #r    = 2
    #norm = ak.to_numpy(ak.sum(p ** r, axis=1))
    #def comp(a, b):
    #    return ak.to_numpy(ak.sum(a*b * p ** (r-2.0), axis=1)) / norm

    #sxx, sxy, sxz = comp(px, px), comp(px, py), comp(px, pz)
    #syy, syz, szz = comp(py, py), comp(py, pz), comp(pz, pz)
    #del px, py, pz, p    # helps memory issue

    #N = len(norm)
    #s = np.zeros((N, 3, 3))
    #s[:, 0, 0], s[:, 0, 1], s[:, 0, 2] = sxx, sxy, sxz
    #s[:, 1, 0], s[:, 1, 1], s[:, 1, 2] = sxy, syy, syz
    #s[:, 2, 0], s[:, 2, 1], s[:, 2, 2] = sxz, syz, szz
    #del sxx, sxy, sxz, syy, syz, szz    # helps memory issue

    #s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.)
    #evals = np.sort(np.linalg.eigvalsh(s), axis=1)
    #return 1.5 * (evals[:, 0] + evals[:, 1])

    ############# Begin boosted mediator frame sphericity
    cut = (events["GenPart_pdgId"]==999999)
    pt   = events["GenPart_pt"][cut]
    eta  = events["GenPart_eta"][cut]
    phi  = events["GenPart_phi"][cut]
    mass = events["GenPart_mass"][cut]
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    #E  = pt*np.cosh(eta)  # treating dark mesons as effectively massless; sqrt(p^2+m^2) if want exact
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    # Mediator 4-vector in lab = sum of meson 4-vectors
    Mx = ak.to_numpy(ak.sum(px, axis=1))
    My = ak.to_numpy(ak.sum(py, axis=1))
    Mz = ak.to_numpy(ak.sum(pz, axis=1))
    ME = ak.to_numpy(ak.sum(E,  axis=1))

    bx, by, bz = Mx/ME, My/ME, Mz/ME
    gamma = 1.0 / np.sqrt(1 - (bx**2 + by**2 + bz**2))
    del Mx, My, Mz, ME

    # Flatten mesons; replicate per-event β across particles via an event index
    n = ak.to_numpy(ak.num(px, axis=1))
    ev = np.repeat(np.arange(len(n)), n)
    px_f = ak.to_numpy(ak.flatten(px)).astype(np.float32)
    py_f = ak.to_numpy(ak.flatten(py)).astype(np.float32)
    pz_f = ak.to_numpy(ak.flatten(pz)).astype(np.float32)
    E_f  = ak.to_numpy(ak.flatten(E)).astype(np.float32)
    del px, py, pz, E
    del pt, eta, phi
    bx_p, by_p, bz_p, g_p = bx[ev], by[ev], bz[ev], gamma[ev]

    # Lorentz boost each meson into the mediator rest frame
    bp = bx_p*px_f + by_p*py_f + bz_p*pz_f
    g2 = g_p**2 / (g_p + 1)
    px_b = px_f + g2*bp*bx_p - g_p*bx_p*E_f
    py_b = py_f + g2*bp*by_p - g_p*by_p*E_f
    pz_b = pz_f + g2*bp*bz_p - g_p*bz_p*E_f

    del px_f, py_f, pz_f, E_f    # helps with memory issue
    del bx_p, by_p, bz_p, g_p
    del bp, g2

    # Sphericity tensor using BOOSTED momenta; per-event sums via bincount
    p2 = px_b**2 + py_b**2 + pz_b**2
    norm = np.bincount(ev, weights=p2)
    def comp(a, b):
        return np.bincount(ev, weights=a*b) / norm
    sxx, sxy, sxz = comp(px_b, px_b), comp(px_b, py_b), comp(px_b, pz_b)
    syy, syz, szz = comp(py_b, py_b), comp(py_b, pz_b), comp(pz_b, pz_b)
    
    del ev
    del px_b, py_b, pz_b    # helps with memory issue
    del p2

    N = len(norm)
    del norm
    s = np.zeros((N, 3, 3))
    s[:, 0, 0], s[:, 0, 1], s[:, 0, 2] = sxx, sxy, sxz
    s[:, 1, 0], s[:, 1, 1], s[:, 1, 2] = sxy, syy, syz
    s[:, 2, 0], s[:, 2, 1], s[:, 2, 2] = sxz, syz, szz
    s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.)
    evals = np.sort(np.linalg.eigvalsh(s), axis=1)
    return 1.5 * (evals[:, 0] + evals[:, 1])


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
    "nPFCands": Plot(
                    name  = "nPFCands",
                    var   = lambda x: x["nPFCands"],
                    bins  = np.arange(200),
                    extra= {
                        "is2D"   : False,
                        "xlabel" : "N_{PF}"
                    }
                ),
   "GenSphericity": Plot(
                    name  = "GenSphericity",
                    var   = GenSphericity,
                    bins  = np.arange(0,1.05, 0.05),
                    extra= {
                        "is2D"   : False,
                        "xlabel" : "S_{#Phi}^{r=2}"
                    }
                ),
}

##############################
# f(T, m) — fitted parameters
##############################

#c_fit = 101.45
#a_fit = 4.81
#A_fit = 1.114
#b_fit = 4.565

c_fit = 100.73
a_fit = 6.40
A_fit = 1.106
b_fit = 4.129

def predict_mu_sigma(T, m):
    mu0 = c_fit / np.sqrt(m**2 + a_fit * T**2)   # = (mu - 0.5), the quantity actually fit
    x = np.log2(T/m)
    g = A_fit * 2**x / np.sqrt(1 + b_fit * 2**(2*x))
    sigma = np.sqrt(mu) * g                      # sigma fit normalized by sqrt(mu)
    mu = mu0 + 0.5                                # add the 0.5 offset back
    return mu, sigma

#################################################################
# GaussianTarget: SUEPSample stand-in whose nPhiGen histogram is
# pre-filled from f(T, m). Used as otherSample in base.reweight.
#################################################################

class GaussianTarget(SUEPSample):
    def collectHistogram(self, cfg, weight=False):
        if cfg.name in self.histos:
            return

    def __init__(self, m, T, cfg, tag, color):
        self.files = []
        self.tag = tag
        self.histos = {}
        self.weights = False
        self.weightFunction = False
        self.weighted_histos = {}
        self.color = color
        self.weighted_color = None
        self.variables = []
        self.clibobject = False
        self.tree = None

        mu, sigma = predict_mu_sigma(T, m)
        h = ROOT.TH1D(tag + cfg.name, tag + cfg.name,
                      len(cfg.bins)-1, array.array('d', cfg.bins))
        #centers = 0.5 * (cfg.bins[1:] + cfg.bins[:-1])    # before
        centers = cfg.bins[:-1]    # after 0.5 left-shift in fitted Nphi in inputs
        for ibin, c in enumerate(centers):
            h.SetBinContent(ibin+1, np.exp(-(c - mu)**2 / (2*sigma**2)))
            h.SetBinError(ibin+1, 0)
        self.histos[cfg.name] = h

#####################
# Configure samples #
#####################

variables = ["nPFCands", "GenPart_pdgId", "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass"]
category = sys.argv[1]
m = float(sys.argv[2])
T = float(sys.argv[3])

# Validated mix (from mix.py)
base = SUEPSample(
    files = joinLists([collectFilesFromFolder(folder) for folder in [
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
    ]]),
    tag = "Mix (Leptonic)",
    color = ROOT.kRed,
    variables = variables,
    weighted_color = ROOT.kBlack)

# Held-out validation point from command line (same style as test.py)
m_str = f"{m:.1f}"
T_str = f"{T:.2f}"
tmpdir = "/tmp/ldishman/suep_closure"

path = f"/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/{category}_{m_str}_{T_str}/*/*/NANOAOD/"

held_out = SUEPSample(
    files = joinLists([collectFilesFromFolder(folder) for folder in glob.glob(path)]),
    tag = f"Target ({category}) m={m_str}, T={T_str}",
    color = ROOT.kBlue,
    variables = variables)

# Predicted Gaussian target from f(T, m) — used as the otherSample to reweight to
gauss_target = GaussianTarget(m, T, allPlots["nPhiGen"],
                              tag = f"f(T,m) pred",
                              color = ROOT.kGreen)

# Tell base (mix) to reweight to the predicted Gaussian (NOT to the actual held-out)
base.weightFunction = partial(base.reweight, otherSample=gauss_target, cfg=allPlots["nPhiGen"])

####################
## Execution area ##
####################

prefix = f"closure_{category}_m{m_str}_T{T_str}_"
outdir = os.path.join(tmpdir, prefix + "output")
os.makedirs(outdir, exist_ok=True)

plot1D = Plotter1D([held_out, base], outdir + "/" + prefix)

root_file = ROOT.TFile.Open(os.path.join(outdir, prefix + "histos.root"), "RECREATE")
for plot in allPlots.values():
    print(psutil.Process(os.getpid()).memory_info().rss/1024**3, "GB")
    histos = plot1D.plotVariable(plot)
    print(psutil.Process(os.getpid()).memory_info().rss/1024**3, "GB")
    for h in histos.values():
        h.Write()
root_file.Close()

# One combined PDF with all three plots side by side
pngs = " ".join(os.path.join(outdir, prefix + n + ".png") for n in allPlots)
os.system(f"montage {pngs} -tile 3x1 -geometry +5+5 {os.path.join(outdir, prefix + 'combined.pdf')}")

#for plot in allPlots:
#    print(psutil.Process(os.getpid()).memory_info().rss/1024**3, "GB")
#    plot1D.plotVariable(allPlots[plot])
#    print(psutil.Process(os.getpid()).memory_info().rss/1024**3, "GB")

Plotter1D([held_out, base, gauss_target], outdir + "/" + prefix + "check_").plotVariable(allPlots["nPhiGen"])
