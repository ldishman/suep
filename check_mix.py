import numpy as np, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Leptonic
# solved event counts (from best run), in same order as labels 
# ORIGINAL:
#events = np.array([6313187, 1951115, 5686671, 3432040, 1958603, 4729473,
#                   96571, 2309574, 1064759, 2533596, 2217663, 2202473, 1967529.0])
# CORRECTED:
#events = np.array([5313187, 3800000, 6486671, 3432040, 2558603, 4729473,
#                   96571, 2309574, 1264759, 2533596, 2217663, 2202473, 1967529.0])
# ALTERNATIVE MIX (BEST?):
#events = np.array([2323200, 2277793, 2713510, 3356519, 3538097, 3574114,
#                   3780015, 3314105, 1710119, 1185562, 1185562])
#events = np.array([3634092, 6499904, 4775033, 5672923, 2558903, 2993450,
#                   2830105, 1939402, 1178572, 2278791, 2236742, 300000.0])

# Generic
# ORIGINAL:
#events = np.array([9635172, 1589036, 1805320, 1722098, 1811623, 1567184,
#                   1520422, 448175, 877815, 793987, 560510.0])
# CORRECT:
events = np.array([1235172, 1689036, 1805320, 1822098, 1611623, 1467184,
                   1420422, 408175, 807815, 923987, 330510.0])

# Hadronic
# ORIGINAL:
#events = np.array([16932465, 2389469, 2550262, 2731531, 2572136, 2365912,
#                   2213570, 1554425, 3313208, 2681531, 38987.0])
# CORRECT:
#events = np.array([1832465, 2900000, 2650262, 2531531, 2872136, 1865912,
#                   1813570, 1095000, 2513208, 2600000, 10000.0])

c_fit, a_fit, A_fit, b_fit = 100.73, 6.40, 1.106, 4.129
def predict_mu_sigma(T, m):
    mu = c_fit/np.sqrt(m**2 + a_fit*T**2) + 0.5
    x  = np.log2(T/m)
    return mu, np.sqrt(mu)*(A_fit*2**x/np.sqrt(1+b_fit*2**(2*x)))

#d = np.load(sorted(glob.glob("mix_output/Leptonic/*shapes.npz"))[0], allow_pickle=True)
d = np.load(sorted(glob.glob("mix_output/Generic/*shapes.npz"))[0], allow_pickle=True)
#d = np.load(sorted(glob.glob("mix_output/Hadronic/*shapes.npz"))[0], allow_pickle=True)
F_s   = d["f_s"]                       # (S, B), same order as events
labels = d["labels"]
bins  = np.arange(F_s.shape[1])
FLOOR = 2
nb    = F_s.T @ events                 # events per N_phi bin (the mix)
for lab, n in zip(labels, events):
    print(f"  {str(lab):18s} {n:>14,.0f}")
print(f"  {'TOTAL':18s} {events.sum():>14,.0f}")
print(f"TOTAL events = {events.sum():,.0f}")
cov   = nb > 0

# SANITY CHECK: target = one sample's own shape -> eps should be ~1/sqrt(N)
si = 4                                    # pick a sample index (any)
ft_self = F_s[si]                         # that sample's own normalized shape
eps_self = np.sqrt((ft_self[cov]**2 / nb[cov]).sum())
print(f"SANITY: target=sample idx{si} ({labels[si]}, N={events[si]:,.0f})")
print(f"  eps = {eps_self*100:.4f}%   vs   1/sqrt(N) = {100/np.sqrt(events[si]):.4f}%")
# exact check: mix of ONLY sample si
nb_solo = F_s[si] * events[si]
good = nb_solo > 0
eps_solo = np.sqrt((F_s[si][good]**2 / nb_solo[good]).sum())
print(f"  solo eps = {eps_solo*100:.4f}%   vs   1/sqrt(N) = {100/np.sqrt(events[si]):.4f}%")

# sweep targets densely and evenly in mu: many masses x many x
mus, epss, params = [], [], []
worst = 0.0
#for mu_target in np.linspace(2.5, 86, 200):        # uniform in Leptonic mu!
for mu_target in np.linspace(2.5, 43, 200):         # uniform in Generic mu!
#for mu_target in np.linspace(2.5, 64, 200):        # uniform in Hadronic mu!
    # x values that keep m in [xlo,8]: invert m = c/((mu-0.5)*sqrt(1+a*2^(2x)))
    # m=8 -> smallest valid 2^(2x); m=xlo -> largest. Solve for x bounds.
    def x_for_m(mm):
        val = (c_fit/(mm*(mu_target-0.5)))**2 - 1   # = a*2^(2x)
        if val <= 0: return None
        return 0.5*np.log2(val/a_fit)
    #x_hi = x_for_m(1.0)    # Leptonic: m=1 gives upper x
    x_hi = x_for_m(2.0)    # Generic: m=2 gives upper x
    #x_hi = x_for_m(1.40)    # Hadronic: m=1.4 gives upper x
    x_lo = x_for_m(8.0)    # m=8 gives lower x
    x_hi = 1.5 if x_hi is None else min(x_hi, 1.5)
    x_lo = -2.0 if x_lo is None else max(x_lo, -2.0)
    if x_lo >= x_hi: continue
    for x in np.linspace(x_lo, x_hi, 40):           # 40 VALID shapes at THIS mu
        m = c_fit / ((mu_target - 0.5) * np.sqrt(1 + a_fit * 2**(2*x)))
        T = m * 2**x
        mu, sig = predict_mu_sigma(T, m)
        g = np.exp(-(bins-mu)**2/(2*sig**2)); g[bins<FLOOR]=0
        if g.sum()==0: continue
        ft = g/g.sum()
        #if ft[~cov].sum() > 0.01: continue
        if ft[~cov].sum() > 0.01:
            if mu > 55:                              # only flag skips near the ceiling
                print(f"  SKIPPED (uncoverable): m={m:.2f} T={T:.2f} mu={mu:.1f}  "
                      f"{100*ft[~cov].sum():.1f}% outside coverage")
            continue
        eps = np.sqrt((ft[cov]**2 / nb[cov]).sum())
        mus.append(mu); epss.append(eps); params.append((m,T))
        worst = max(worst, eps)
mus  = np.array(mus); epss = np.array(epss)
hi = np.argmax(mus)
print(f"highest-mu target tested: mu={mus[hi]:.1f}  eps={epss[hi]*100:.4f}%")
wi = np.argmax(epss)
m, T = params[wi]
mu, sig = predict_mu_sigma(T, m)
g = np.exp(-(bins-mu)**2/(2*sig**2)); g[bins<FLOOR]=0
ft = g/g.sum()
contrib = ft[cov]**2 / nb[cov]
cb = np.arange(F_s.shape[1])[cov]
print(f"\nWorst: m={m:.2f} T={T:.2f}  eps={epss[wi]*100:.4f}%")
for b in np.argsort(contrib)[::-1][:5]:
    nphi = cb[b]
    top = np.argmax(F_s[:, nphi] * events)
    print(f"  N_phi={nphi}: {contrib[b]/epss[wi]**2*100:4.1f}%  top sample=idx{top} ({events[top]:,.0f} ev)")
print(f"WORST eps = {worst*100:.4f}%  over {len(mus)} dense targets "
      f"(mu range {mus.min():.1f}--{mus.max():.1f})")

# plot: dense cloud + worst-case envelope 
o = np.argsort(mus)
plt.figure(figsize=(8,6))
plt.plot(mus[o], epss[o]*100, ".", ms=2, alpha=0.3)
edges = np.linspace(mus.min(), mus.max(), 70)
k  = np.digitize(mus, edges); mc = 0.5*(edges[:-1]+edges[1:])
env = np.array([epss[k==j].max() if np.any(k==j) else np.nan for j in range(1, len(edges))])
plt.plot(mc, env*100, "b-", lw=1.5, label="worst-case ε(μ)")
plt.axhline(0.1, color="r", ls="--", label="target 0.1%")
plt.xlabel(r"$\mu$ (target mean $N_\phi$)"); plt.ylabel(r"$\epsilon$ [%]")
#plt.title("Leptonic: ε audit on DENSE μ-uniform grid")
plt.title("Generic: ε audit on DENSE μ-uniform grid")
#plt.title("Hadronic: ε audit on DENSE μ-uniform grid")
plt.legend()
#plt.savefig("mix_output/leptonic_dense_audit.png", dpi=120, bbox_inches="tight")
plt.savefig("mix_output/generic_dense_audit.pdf", dpi=120, bbox_inches="tight")
#plt.savefig("mix_output/hadronic_dense_audit.png", dpi=120, bbox_inches="tight")
print("saved epsilon plot")

# stacked coverage plot with the (tweaked) event counts
centers = np.arange(F_s.shape[1])[cov]
stack   = events[:, None] * F_s[:, cov]
plt.figure(figsize=(8,6))
plt.stackplot(centers, stack, labels=[str(l) for l in labels])
plt.plot(centers, stack.sum(0), "k-", lw=1.2)
plt.xlabel(r"$N_\phi$"); plt.ylabel(r"mix events per bin  $n_b=\sum_s N_s f_s$")
plt.ticklabel_format(axis="y", style="sci", scilimits=(6,6))
#plt.title("Leptonic: Optimized mix coverage (tweaked)")
plt.title("Generic: Optimized mix coverage (tweaked)")
#plt.title("Hadronic: Optimized mix coverage (tweaked)")
plt.legend(fontsize=7, ncol=2)
#plt.savefig("mix_output/leptonic_coverage_tweaked.png", dpi=120, bbox_inches="tight")
plt.savefig("mix_output/generic_coverage_tweaked.pdf", dpi=120, bbox_inches="tight")
#plt.savefig("mix_output/hadronic_coverage_tweaked.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved coverage plot")
