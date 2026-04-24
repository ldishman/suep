from interpFromScan_uproot import *
import awkward as ak
import os, sys, glob
from functools import partial
import ROOT

###################
# Configure plots #
###################

# These are functions used to define variables on the fly #
def MeanPhiPt(events):
    pt = events["GenPart_pt"]
    pt = pt[events["GenPart_pdgId"]==999999]
    return ak.sum(pt, axis=1)/ak.num(pt, axis=1)


def MasslessInvMass(events):
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

    totpx = ak.sum(px, axis=1)
    totpy = ak.sum(py, axis=1)
    totpz = ak.sum(pz, axis=1)
    totp  = ak.sum(p, axis=1)
    return np.sqrt(totp**2 - totpz**2 - totpy**2-totpx**2)

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
    norm = np.squeeze(ak.sum(p ** r, axis=1, keepdims=True))
    s = np.array([[
                       ak.sum(px*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(py*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(pz*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                       ]])
    s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
    s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.) 
    evals = np.sort(np.linalg.eigvals(s))
    return np.real(1.5*(evals[:,0] + evals[:,1]))

# Define plots to do, one entry means one comparison plot
# Note that if we use more variables they need to be added to the "variables" list below (this is a trick to not load the whole set of variables in the files)
allPlots = {
    "nPhiGen": Plot(
                    name  = "nPhiGen", # Internal name and how the file will be saved
                    var   = lambda x: ak.sum(x["GenPart_pdgId"] == 999999, axis=1), # Definition of this variable (has to be a function)
                    bins  = np.arange(200), # Binning edges
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
#"""    "MasslessInvMass" : Plot(
#                    name = "MasslessInvMass",
#                    var  = MasslessInvMass,
#                    bins = np.arange(0, 250, 2),
#                    extra= {
#                        "is2D"   : False,
#                        "xlabel" : "M_{\sum{\phi}} [GeV]"
#                    }
#                ),
#
#    "GenSphericity": Plot(
#                    name  = "GenSphericity",
#                    var   = GenSphericity,
#                    bins  = np.arange(0,1.05, 0.05),
#                    extra= {
#                        "is2D"   : False,
#                        "xlabel" : "S_{#Phi}^{r=2}"
#                    }
#                ), # This crashes due to lack of memory...
#    "MeanPhiPt": Plot(
#                    name = "MeanPhiPt",
#                    var  = MeanPhiPt,
#                    bins = np.arange(0, 50, 0.5),
#                    extra= {
#                        "is2D"   : False,
#                        "xlabel" : "Mean p_{T}^{#Phi} [GeV]"
#                    }
#                ),
#    "SumPhiPt": Plot(
#                    name = "SumPhiPt",
#                    var  = lambda x: ak.sum(x["GenPart_pt"]*(x["GenPart_pdgId"] == 999999), axis=1),
#                    bins = np.concatenate([np.arange(0, 200, 10),np.arange(200, 525, 50)]),
#                    extra= {
#                        "is2D"   : False,
#                        "xlabel" : "#sum p_{T}^{#Phi} [GeV]"
#                    }
#                ),
#    "2D_nPhiGen_SumPhiPt" : Plot(
#                    name = "2D_nPhiGen_SumPhiPt",
#                    var  = None,
#                    bins = [np.arange(200), np.concatenate([np.arange(0, 200, 10),np.arange(200, 525, 50)])],
#                    extra= {
#                        "varX"   : lambda x: ak.sum(x["GenPart_pdgId"] == 999999, axis=1),
#                        "varY"   : lambda x: ak.sum(x["GenPart_pt"]*(x["GenPart_pdgId"] == 999999), axis=1),
#                        "is2D"   : True,
#                        "xlabel" : "#sum p_{T}^{#Phi} [GeV]"
#                    }
#                ),"""
}

#####################
# Configure samples #
#####################

# These will be loaded from the trees 
variables = ["nPFCands", "GenPart_pdgId", "GenPart_pt", "GenPart_eta", "GenPart_phi"]#, "GenPart_mass"]

base   = SUEPSample(
        files = joinLists([collectFilesFromFolder(folder) for folder in [ # Files to load
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_0.75/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T0.75_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_1.06/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T1.06_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_1.50/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T1.50_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_2.12/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T2.12_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_3.00/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T3.00_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_4.24/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T4.24_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_6.00/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T6.00_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_8.49/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T8.49_HT-1_/NANOAOD/",
            "/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/Leptonic_3.0_12.00/UL18/ZHleptonicpythia_leptonic_M125_MD3.0_T12.00_HT-1_/NANOAOD/",
            ]]),
        tag = "Mix", #Name in the legend
        color = ROOT.kRed, # Color for nominal
        variables = variables, #Which variables will need to be loaded from root
        weighted_color= ROOT.kBlack) # Color for weighted version

category = sys.argv[1]
m = float(sys.argv[2])
T = float(sys.argv[3])
m_str = f"{m:.1f}"
T_str = f"{T:.2f}"
tmpdir = "/tmp/ldishman/suep_job"

path = f"/eos/cms/store/group/phys_exotica/SUEPs/ZH_GenFixed_Samples/2018/{category}_{m_str}_{T_str}/*/*/NANOAOD/"

target = SUEPSample(
        files = joinLists([collectFilesFromFolder(folder) for folder in glob.glob(path)]),
        tag = f"m={m_str}, T={T_str}",
        color = ROOT.kBlue,
        variables = variables)

# This means that the base samples will optionally be weighted by using the reweight method of the SUEPSample class, which takes other sample and a plot configuration as inputs
# If you look at SUEPSample.reweight in interpFromScan_uproot.py, this means that it will make the histograms of the plot configured as "nPhiGen" for both samples, then weight based on the ratio between the two such that "base" becomes "target"
base.weightFunction = partial(base.reweight, otherSample=target, cfg = allPlots["nPhiGen"])

####################
## Execution area ##
####################
# This is a simple plotter 
prefix = f"{category}_m{m_str}_T{T_str}_"
#outdir = f"/afs/cern.ch/user/l/ldishman/suep/m3_output/{prefix}output/"
outdir = os.path.join(tmpdir, prefix + "output")

# Create the directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

plot1D = Plotter1D([target, base], outdir + "/" + prefix)
#print(dir(plot1D))

# Save ROOT file of histograms
root_file = ROOT.TFile.Open(os.path.join(outdir, prefix + "histos.root"), "RECREATE")
for plot in allPlots.values():
    histos = plot1D.plotVariable(plot)  # returns dict
    for h in histos.values():
        h.Write()
root_file.Close()

# And execute the plotter for all plots
for plot in allPlots:
    plot1D.plotVariable(allPlots[plot])

