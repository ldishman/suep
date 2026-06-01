from interpFromScan_uproot import *
import awkward as ak
import os, sys, glob
from functools import partial
import ROOT
import numpy as np
import array

###################
# Configure plots #
##################

def GenSphericity(events):
    cut = (events["GenPart_pdgId"]==999999)
    # Only the dark mesons
    pt   = events["GenPart_pt"][cut]
    eta  = events["GenPart_eta"][cut]
    phi  = events["GenPart_phi"][cut]
    # Convert to euclidean
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    p  = pt*np.cosh(eta)

    # Sphericity magic
    r    = 2
    #norm = np.squeeze(ak.sum(p ** r, axis=1, keepdims=True))
    #s = np.array([[
    #                   ak.sum(px*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(px*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(px*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
    #                  ],
    #                  [
    #                   ak.sum(py*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(py*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(py*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
    #                  ],
    #                  [
    #                   ak.sum(pz*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(pz*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
    #                   ak.sum(pz*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
    #                   ]])
    #s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
    #s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.)
    #evals = np.sort(np.linalg.eigvals(s))
    #return np.real(1.5*(evals[:,0] + evals[:,1]))

    norm = ak.to_numpy(ak.sum(p ** r, axis=1))
    def comp(a, b):
        return ak.to_numpy(ak.sum(a*b * p ** (r-2.0), axis=1)) / norm

    sxx, sxy, sxz = comp(px, px), comp(px, py), comp(px, pz)
    syy, syz, szz = comp(py, py), comp(py, pz), comp(pz, pz)

    N = len(norm)
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

c_fit = 101.45
a_fit = 4.81
A_fit = 1.114
b_fit = 4.565

def predict_mu_sigma(T, m):
    mu = c_fit / np.sqrt(m**2 + a_fit * T**2)
    x = np.log2(T/m)
    g = A_fit * 2**x / np.sqrt(1 + b_fit * 2**(2*x))
    sigma = np.sqrt(mu) * g
    return mu, sigma

#################################################################
# GaussianTarget: SUEPSample stand-in whose nPhiGen histogram is
# pre-filled from f(T, m). Used as otherSample in base.reweight.
#################################################################

class GaussianTarget(SUEPSample):
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
        centers = 0.5 * (cfg.bins[1:] + cfg.bins[:-1])
        for ibin, c in enumerate(centers):
            h.SetBinContent(ibin+1, np.exp(-(c - mu)**2 / (2*sigma**2)))
            h.SetBinError(ibin+1, 0)
        self.histos[cfg.name] = h

#####################
# Configure samples #
#####################

variables = ["nPFCands", "GenPart_pdgId", "GenPart_pt", "GenPart_eta", "GenPart_phi"]
category = sys.argv[1]
m = float(sys.argv[2])
T = float(sys.argv[3])

# Validated mix (from mix.py)
base = SUEPSample(
    files = joinLists([collectFilesFromFolder(folder) for folder in [
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T1.00_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_1.0_2.83/UL18/ZHleptonicpythia_leptonic_M125_MD1.0_T2.83_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_2.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD2.0_T1.00_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_1.06/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T1.06_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_4.0_1.00/UL18/ZHleptonicpythia_leptonic_M125_MD4.0_T1.00_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_6.0_3.00/UL18/ZHleptonicpythia_leptonic_M125_MD6.0_T3.00_HT-1_/NANOAOD/",
        "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_8.0_16.00/UL18/ZHleptonicpythia_leptonic_M125_MD8.0_T16.00_HT-1_/NANOAOD/",
    ]]),
    tag = f"Mix ({category})",
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
    histos = plot1D.plotVariable(plot)
    for h in histos.values():
        h.Write()
root_file.Close()

for plot in allPlots:
    plot1D.plotVariable(allPlots[plot])
