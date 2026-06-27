import ROOT
import os
import uproot
import numpy as np
import array
import correctionlib.convert as conv
#import awkward as ak
ROOT.gROOT.SetBatch(True)

ROOT.gStyle.SetOptStat(0)

def joinLists(lists):
    out = []
    for l in lists:
        out = out + l
    return out

def collectFilesFromFolder(folder, filt="root"):
    out = []
    for f in os.listdir(folder):
        if not(filt in f): continue
        out.append(folder +"/"+ f)
    return out

class Plot(object):
    def __init__(self, name, var, bins, extra= {}):
        self.name     = name
        self.var      = var
        self.bins     = bins
        self.is2D     = False
        self.xlabel   = ""
        self.ylabel   = ""
        for e in extra:
            setattr(self, e, extra[e])

class SUEPSample(object):
    def __init__(self, files, tag, color, variables, weighted_color=None):
        self.files  = files
        self.tag    = tag
        self.histos = {}
        self.weights = False
        self.weightFunction = False
        self.weighted_histos = {}
        self.color = color
        self.weighted_color = weighted_color
        self.variables = variables
        self.clibobject = False
        self.getFiles()

    def getFiles(self):
        print("Load trees for sample %s"%self.tag)
        self.tree = uproot.concatenate(["%s:Events"%f for f in self.files], self.variables, num_workers=8)
                
    def getHistogram(self, cfg, weight=False):
        if weight:
            if not(cfg.name in self.weighted_histos):
                self.collectHistogram(cfg, weight)
            return self.weighted_histos[cfg.name]
        else:
            if not(cfg.name in self.histos):
                self.collectHistogram(cfg, weight)
            return self.histos[cfg.name]

    def getWeights(self, n=1):
        if type(self.weights) != type(False):
            return self.weights**n
        else:
            print("Updating weights for sample %s"%self.tag)
            self.computeWeights()
            return self.weights**n

    def computeWeights(self):
        self.weights = self.weightFunction(self.tree)
    
    def reweight(self, tree, otherSample, cfg):
        if not self.clibobject:
            hDen = self.getHistogram(cfg, False)
            hNum = otherSample.getHistogram(cfg, False)
            #hQuo = hNum.Clone(hNum.GetName() + "_over_" + hDen.GetName())
            #hQuo.Divide(hDen)
            #hQuoUpRoot = uproot.pyroot.from_pyroot(hQuo)

            # Convert contents to numpy arrays
            num_vals = np.array([hNum.GetBinContent(i+1) for i in range(hNum.GetNbinsX())])
            den_vals = np.array([hDen.GetBinContent(i+1) for i in range(hDen.GetNbinsX())])
            edges = np.array([hDen.GetBinLowEdge(i+1) for i in range(hDen.GetNbinsX())] +
                         [hDen.GetBinLowEdge(hDen.GetNbinsX()) + hDen.GetBinWidth(hDen.GetNbinsX())])

            # Safe division: avoid divide by zero
            ratio = np.divide(num_vals, den_vals, out=np.ones_like(num_vals), where=den_vals!=0)

            #self.clibobject = conv.from_histogram(hQuoUpRoot)
            #self.clibobject.data.flow = "clamp"
            #self.clibobject = self.clibobject.to_evaluator()

            # Build a boost_histogram and convert to correctionlib histogram
            import boost_histogram as bh
            bh_hist = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
            #bh_hist.view(flow=True)[:] = ratio
            #bh_hist.view(flow=True)[:] = np.zeros(bh_hist.size, dtype=[('value','f8'), ('variance','f8')])
            bh_hist.view(flow=False)['value'] = ratio
            self.clibobject = conv.from_histogram(bh_hist).to_evaluator()
            #self.clibobject.data.flow = "clamp"

            #self.clibobject = conv.from_numpy(edges, ratio, flow="clamp").to_evaluator()

            #self.clibobject = conv.from_histogram({"values": ratio, "edges": [edges]}).to_evaluator()
            #self.clibobject.data.flow = "clamp"

        if cfg.is2D:
            return self.clibobject.evaluate(cfg.varX(tree), cfg.varY(tree))
        else:
            return self.clibobject.evaluate(cfg.var(tree))

    def collectHistogram(self, cfg, weight=False):
        if cfg.is2D:
            self.collectHistogram2D(cfg, weight)
        else:
            self.collectHistogram1D(cfg, weight)

    def collectHistogram1D(self, cfg, weight=False):
        print("Build%s 1D histogram %s for %s"%(" weighted" if weight else "", cfg.name, self.tag))
        if weight:
            #theVar = cfg.var(self.tree)
            #theVar = ak.to_numpy(ak.flatten(theVar)) # flatten array for sqrt use
            #theVar = np.array(cfg.var(self.tree), dtype=object) # to allow sqrt a different way
            theVar = np.concatenate([np.atleast_1d(x) for x in cfg.var(self.tree)])   # flatten list-of-arrays

            #w = ak.to_numpy(ak.flatten(self.getWeights())) # same
            #w = np.array(self.getWeights(), dtype=object) # same
            w = np.concatenate([np.atleast_1d(x) for x in self.getWeights()])

            #yields, edges = np.histogram(theVar, cfg.bins, weights=self.getWeights()) 
            #errors, edges = np.sqrt(np.histogram(theVar, cfg.bins, weights=self.getWeights(2)))            
            #yields, edges = np.histogram(theVar, cfg.bins, weights=np.asarray(w, dtype=float))
            #hist_sq, edges = np.histogram(theVar, cfg.bins, weights=np.asarray(w, dtype=float)**2)
            #errors = np.sqrt(hist_sq)
            yields, edges = np.histogram(theVar, cfg.bins, weights=w)
            hist_sq, edges = np.histogram(theVar, cfg.bins, weights=w**2)
            errors = np.sqrt(hist_sq)
        else:
            yields, edges = np.histogram(cfg.var(self.tree), cfg.bins)
            errors = yields**0.5

        tmp = ROOT.TH1D(self.tag + cfg.name + "_weights" if weight else "", self.tag + cfg.name + "_weights" if weight else "", len(cfg.bins)-1, array.array('d', cfg.bins))
        for ibin in range(len(yields)):
            tmp.SetBinContent(ibin+1, yields[ibin])
            tmp.SetBinError(ibin+1, errors[ibin])

        if weight:
            self.weighted_histos[cfg.name] = tmp
        else:
            self.histos[cfg.name] = tmp

    def collectHistogram2D(self, cfg, weight=False):
        print("Build%s 2D histogram %s for %s"%(" weighted" if weight else "", cfg.name, self.tag))
        varX = cfg.varX(self.tree)
        varY = cfg.varY(self.tree) 
        if weight:
            yields, xedges, yedges = np.histogram2d(varX, varY, cfg.bins, weights=self.getWeights())
            errors, xedges, yedges = np.sqrt(np.histogram2d(varX, varY, cfg.bins, weights=self.getWeights(2)))
        else:
            yields, xedges, yedges = np.histogram2d(varX, varY, cfg.bins)
            errors = yields**0.5

        tmp = ROOT.TH2D(self.tag + cfg.name + "_weights" if weight else "", self.tag + cfg.name + "_weights" if weight else "", len(cfg.bins[0])-1, array.array('d', cfg.bins[0]), len(cfg.bins[1])-1, array.array('d', cfg.bins[1]))
        for ibin in range(len(yields)):
            for jbin in range(len(yields[ibin])):
                tmp.SetBinContent(ibin+1, jbin+1, yields[ibin][jbin])
                tmp.SetBinError(ibin+1, jbin+1, errors[ibin][jbin])
        if weight:
            self.weighted_histos[cfg.name] = tmp
        else:
            self.histos[cfg.name] = tmp


class Plotter1D(object):
    def __init__(self, samples, output):
        self.samples = samples
        self.n = len(self.samples)
        self.output = output

    def plotVariable(self, cfg):
        print("Now plotting %s"%cfg.name)
        c = ROOT.TCanvas("c"+ cfg.name, "c"+cfg.name, 800, 900)
        p1 = ROOT.TPad("p1"+cfg.name, "p1"+cfg.name, 0., 0.25, 1., 1.)
        p1.SetLogy()
        p2 = ROOT.TPad("p2"+cfg.name, "p2"+cfg.name, 0., 0., 1., 0.25)
        p1.SetBottomMargin(0.05)
        p1.SetRightMargin(0.04)
        p2.SetRightMargin(0.04)
        p1.SetLeftMargin(0.15)
        p2.SetLeftMargin(0.15)
        p2.SetTopMargin(0.05)
        p2.SetBottomMargin(0.33)
        p1.Draw()
        p2.Draw()
        p1.cd()
        #tl = ROOT.TLegend(0.6,0.6, 0.9, 0.9)
        tl = ROOT.TLegend(0.58, 0.74, 0.9, 0.89)   # shorter box -> entries packed closer
        tl.SetTextSize(0.028)
        tl.SetBorderSize(0)
        tl.SetFillStyle(0)                          # transparent -> curves show through
        histos = {}
        raw = {}                 # raw (pre-normalization) counts+errors for chi2
        first = True
        sampletags = []
        for sample in self.samples:
            sampletags.append(sample.tag)
            sample.collectHistogram(cfg, False)
            histos[sample.tag ] = sample.histos[cfg.name].Clone(cfg.name + "_" + sample.tag + "_total")
            histos[sample.tag ].SetDirectory(0)
            histos[sample.tag ].SetLineColor(sample.color)
            histos[sample.tag ].SetFillColor(0)
            nbx = histos[sample.tag].GetNbinsX()
            raw[sample.tag] = ([histos[sample.tag].GetBinContent(b) for b in range(1, nbx+1)],
                               [histos[sample.tag].GetBinError(b)   for b in range(1, nbx+1)])
            histos[sample.tag ].Scale(1./histos[sample.tag].Integral())
            if sample.weightFunction:
                newtag = "Weighted %s"%sample.tag.split(" (")[0]   # "Weighted Mix (Leptonic)" -> "Weighted Mix"
                sampletags.append(newtag)
                sample.collectHistogram(cfg, True)
                histos[newtag] = sample.weighted_histos[cfg.name].Clone(cfg.name + "_" + sample.tag + "weighted_total")
                histos[newtag].SetDirectory(0)
                histos[newtag].SetLineColor(sample.weighted_color)
                histos[newtag].SetFillColor(0)
                #histos[newtag].Scale(1./histos[newtag].Integral())
                nbw = histos[newtag].GetNbinsX()
                raw[newtag] = ([histos[newtag].GetBinContent(b) for b in range(1, nbw+1)],
                               [histos[newtag].GetBinError(b)   for b in range(1, nbw+1)])
                integral = histos[newtag].Integral()
                if integral != 0:
                    histos[newtag].Scale(1./integral)


        for tag in sampletags:
            if first:
                histos[tag ].SetMaximum(1)
                histos[tag ].SetMinimum(0.00001)
                histos[tag ].SetTitle(";;Normalized events")
                histos[tag ].Draw("hist")
                first = False
            else:
                histos[tag ].Draw("histsame")
            tl.AddEntry(histos[tag ], tag, "l")
        tl.Draw("same")
        p2.cd()
        ratios = {}
        for tag in sampletags:
            ratios[tag] = histos[tag].Clone(histos[tag].GetName() + "_ratio")
            #ratios[tag].SetLineColor()
            ratios[tag].Divide(histos[sampletags[0]])
            #ratios[tag].SetTitle(";%s;X/%s"%(cfg.xlabel, sampletags[0]))
            ratios[tag].SetTitle(";%s;X / Target"%cfg.xlabel)
            ratios[tag].SetMaximum(2)
            ratios[tag].SetMinimum(0)
            ratios[tag].GetXaxis().SetTitleSize(0.16)
            ratios[tag].GetYaxis().SetTitleSize(0.07)
            ratios[tag].GetXaxis().SetTitleOffset(0.8)
            ratios[tag].GetYaxis().SetTitleOffset(0.5)
            ratios[tag].GetXaxis().SetLabelSize(0.1)
            ratios[tag].GetYaxis().SetLabelSize(0.06)
            ratios[tag].Draw("psame")
        # goodness of fit: total variation distance between normalized shapes
        # 0 = identical, ~0.05 = excellent, 0.2 = visibly off, 1 = disjoint
        p1.cd()
        obs_c, _ = raw[sampletags[0]]                       # held-out = first sample
        obs_tot  = sum(obs_c)
        gof_text = ROOT.TLatex(); gof_text.SetNDC(True); gof_text.SetTextSize(0.032)
        ypos = 0.86
        for tag in sampletags[1:]:                          # mix (unweighted) and Weighted mix
            pred_c, _ = raw[tag]
            ptot = sum(pred_c)
            if ptot > 0 and obs_tot > 0:
                tvd = 0.5 * sum(abs(p/ptot - o/obs_tot) for p, o in zip(pred_c, obs_c))
            else:
                tvd = float("nan")
            gof_text.DrawLatex(0.18, ypos, "TVD (%s) = %.3f" % (tag, tvd))
            ypos -= 0.05
        #print(f"raw should be large but it's: {sum(raw[sampletags[0]][0])}")
        c.SaveAs(self.output + cfg.name +".pdf")
        c.SaveAs(self.output + cfg.name +".png")   
        c.Close()
        del c
        return histos

