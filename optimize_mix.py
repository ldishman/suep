import glob
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps_target   = 0.001
eps_solve    = 0.00095   # solve to a smaller epsilon (if need) to leave headroom to actual epsilon requirement
FLAVOR = "Leptonic" 
#files = sorted(glob.glob("mix_output/**/*shapes.npz", recursive=True))
files = sorted(glob.glob(f"mix_output/{FLAVOR}/*shapes.npz"))
assert len(files) == 1, f"expected 1 {FLAVOR} shapes file, found {len(files)}: {files}"
print("USING:", files[0])

d0     = np.load(files[0], allow_pickle=True)
F_s    = d0["f_s"]                                                            # (S, B) sample pdfs (identical in every file)
print((np.arange(F_s.shape[1])[F_s.sum(0) > 0]).min())   # Use this to check what FLOOR should be (min Nphi)
labels = d0["labels"]
#F_t    = np.array([np.load(f, allow_pickle=True)["f_target"] for f in files])  # (T, B)
#names  = [str(np.load(f, allow_pickle=True)["target"]) for f in files]
c_fit, a_fit = 100.73, 6.40
A_fit, b_fit = 1.106, 4.129
def predict_mu_sigma(T, m):
    mu = c_fit / np.sqrt(m**2 + a_fit * T**2) + 0.5
    x  = np.log2(T / m)
    return mu, np.sqrt(mu) * (A_fit * 2**x / np.sqrt(1 + b_fit * 2**(2*x)))
M_LO = {"Leptonic": 1.0, "Generic": 2.0, "Hadronic": 1.40}[FLAVOR]
M_HI, NM = 8.0, 25    
X_LO, X_HI, NX = -2.0, 1.5, 36     # stop at x=1.5; x=2 column is where the σ fit breaks (see fit.py mask)
bins = np.arange(F_s.shape[1])
F_t, names = [], []

#FLOOR = 2 if FLAVOR in ("Generic", "Hadronic") else 1
FLOOR = 2
for m in np.linspace(M_LO, M_HI, NM):
    for x in np.linspace(X_LO, X_HI, NX):
        T = m * 2**x
        mu, sigma = predict_mu_sigma(T, m)
        g = np.exp(-(bins - mu)**2 / (2*sigma**2))
        g[bins < FLOOR] = 0.0               # kill the unphysical sub-floor tail
        F_t.append(g / g.sum())             # renormalize over what remains
        names.append(f"m={m:.2f}, T={T:.2f}")
F_t = np.array(F_t)

cover = F_s.sum(axis=0) > 0          # drop N_phi bins no sample covers (uncoverable)
lost = F_t[:, ~cover].sum(axis=1)                 # each target's pdf mass in uncoverable bins
for i in np.where(lost > 1e-6)[0]:
    print(f"WARNING: {names[i]} has {100*lost[i]:.2f}% of its N_phi outside mix coverage (uncoverable)")
KEEP = lost < 0.01

# Print bounds of method's success
# --- quantify the uncoverable region ---
mu_grid = np.array([predict_mu_sigma(m*2**x, m)[0]
                    for m in np.linspace(M_LO, M_HI, NM)
                    for x in np.linspace(X_LO, X_HI, NX)])   # same order as F_t/names

print(f"\n=== COVERAGE REPORT ===")
print(f"grid targets total:      {len(KEEP)}")
print(f"dropped (>1% uncov.):    {(~KEEP).sum()}  ({100*(~KEEP).mean():.1f}%)")
print(f"kept (guaranteed):       {KEEP.sum()}")
print(f"guaranteed mu range:     {mu_grid[KEEP].min():.1f} -- {mu_grid[KEEP].max():.1f}")
print(f"mix physical ceiling:    mu = {np.arange(F_s.shape[1])[cover].max():.0f}  (highest covered N_phi bin)")
# the (m,T) boundary of what's dropped, per mass
print("dropped region per mass (min T that becomes uncoverable):")
ms = np.repeat(np.linspace(M_LO, M_HI, NM), NX)
Ts = np.array([m*2**x for m in np.linspace(M_LO, M_HI, NM)
                       for x in np.linspace(X_LO, X_HI, NX)])
for mm in np.unique(ms):
    dropped_T = Ts[(ms == mm) & ~KEEP]
    if len(dropped_T):
        print(f"  m={mm:.2f}:  T >= {dropped_T.min():.2f} uncoverable")
    else:
        print(f"  m={mm:.2f}:  fully covered")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05]:
    k = lost < thr
    print(f"  tol {thr*100:.1f}%:  {k.sum()}/{len(k)} kept, "
          f"mu_max = {mu_grid[k].max():.1f}")
peak_bin = mu_grid                      # approx peak location
wall_loss  = (~KEEP) & (peak_bin < 5)   # low-mu, hitting N_phi=0 wall
reach_loss = (~KEEP) & (peak_bin > 40)  # high-mu, past mix ceiling
print(f"dropped by N_phi=0 wall (model breakdown): {wall_loss.sum()}")
print(f"dropped by mix ceiling (no sample reaches): {reach_loss.sum()}")

# Trim
F_t   = F_t[KEEP]
names = [n for n, k in zip(names, KEEP) if k]
centers = np.arange(F_s.shape[1])[cover]        # N_phi value of each kept bin
F_s, F_t = F_s[:, cover], F_t[:, cover]

S    = F_s.shape[0]
T    = F_t.shape[0]
SCALE  = 1e6
N    = cp.Variable(S, nonneg=True)                      # events needed per sample, now in units of 1e6 events
nb   = F_s.T @ N                                        # mix events per bin
inv_nb = cp.inv_pos(nb)
#cons = [cp.sum(cp.multiply(F_t[t]**2, cp.inv_pos(nb))) <= eps**2 for t in range(len(F_t))]    # the constraint
#cons   = [cp.sum(F_t[t]**2 * inv_nb) <= (eps**2) * SCALE for t in range(T)]   # 1e-6 * 1e6 = 1
cons = [cp.sum(cp.multiply(F_t[t]**2, inv_nb)) <= (eps_solve**2) * SCALE for t in range(T)]
cons += [N >= 0.1]    # minimize events with a per-sample floor
prob = cp.Problem(cp.Minimize(cp.sum(N)), cons)    # the problem = minimize total N under uncertainty constraint
#prob = cp.Problem(cp.Minimize(0), cons)    # just drop the N minimization and find A solution for epsilon req.
#prob.solve(solver=cp.CLARABEL, max_iter=500000)
prob.solve(solver=cp.SCS, max_iters=500000, eps=1e-4, verbose=True)
events = N.value * SCALE                                    # back to real counts

print("status:", prob.status)
for lab, n in zip(labels, events):
    print(f"  {str(lab):18s} {n:>14,.0f}")
print(f"  {'TOTAL':18s} {events.sum():>14,.0f}   ({len(F_t)} targets, eps={eps_solve})")

# stacked mix coverage with the optimized counts; envelope should be ~flat over target range
stack = events[:, None] * F_s                  # (S, B): events per N_phi bin from each sample
plt.stackplot(centers, stack, labels=[str(l) for l in labels])
plt.plot(centers, stack.sum(0), "k-", lw=1.2)   # the "flat line" = total n_b
plt.xlabel(r"$N_\phi$"); plt.ylabel(r"mix events per bin  $n_b=\sum_s N_s f_s$")
plt.title("Optimized mix coverage")
plt.legend(fontsize=7, ncol=2)
plt.savefig("mix_output/mix_coverage.pdf"); plt.close()

# per-target uncertainty from the optimized mix — doubles as the feasibility check
#nb_opt = F_s.T @ events                              # real events per covered bin
#eps_t  = np.sqrt((F_t**2 / nb_opt).sum(axis=1))      # ε_t for each target
#order  = np.argsort(eps_t)
#mu_t = (F_t * centers).sum(axis=1) / F_t.sum(axis=1)   # mean N_phi of each target
#plt.figure()
#plt.plot(mu_t, eps_t*100, "o", ms=3)
#plt.axhline(eps*100, color="r", ls="--", label=f"target {eps*100:.1f}%")
#plt.xlabel(r"$N_\phi$ (target mean)"); plt.ylabel(r"$\epsilon_t$ [%]")
#plt.title("Per-target stat. uncertainty at optimized mix"); plt.legend()
#plt.savefig("mix_output/epsilon_per_target.pdf"); plt.close()
#print(f"max ε_t = {eps_t.max()*100:.4f}%  at  {names[eps_t.argmax()]}")

# Smoothed version of above rather than direct plot over existing sample points
nb_opt = F_s.T @ events
eps_t  = np.sqrt((F_t**2 / nb_opt).sum(axis=1))
mu_t   = (F_t * centers).sum(axis=1) / F_t.sum(axis=1)
o = np.argsort(mu_t)
plt.figure()
plt.plot(mu_t[o], eps_t[o]*100, ".", ms=2, alpha=0.3)        # dense grid cloud
edges = np.linspace(mu_t.min(), mu_t.max(), 60)               # worst-case envelope = guarantee
k  = np.digitize(mu_t, edges); mc = 0.5*(edges[:-1]+edges[1:])
env = np.array([eps_t[k==j].max() if np.any(k==j) else np.nan for j in range(1, len(edges))])
plt.plot(mc, env*100, "b-", lw=1.5, label="worst-case ε(μ)")
plt.axhline(eps_target*100, color="r", ls="--", label=f"target {eps_target*100:.1f}%")
plt.xlabel(r"$\mu$ (target mean $N_\phi$)"); plt.ylabel(r"$\epsilon$ [%]")
plt.title("ε across full (m,T) plane"); plt.legend()
plt.savefig("mix_output/epsilon_envelope.pdf"); plt.close()
print(f"max ε = {eps_t.max()*100:.4f}%  at  {names[eps_t.argmax()]}")
