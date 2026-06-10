import glob
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps   = 0.001
files = sorted(glob.glob("mix_output/**/*shapes.npz", recursive=True))

d0     = np.load(files[0], allow_pickle=True)
F_s    = d0["f_s"]                                                            # (S, B) sample pdfs (identical in every file)
labels = d0["labels"]
F_t    = np.array([np.load(f, allow_pickle=True)["f_target"] for f in files])  # (T, B)
names  = [str(np.load(f, allow_pickle=True)["target"]) for f in files]

cover = F_s.sum(axis=0) > 0          # drop N_phi bins no sample covers (uncoverable)
lost = F_t[:, ~cover].sum(axis=1)                 # each target's pdf mass in uncoverable bins
for i in np.where(lost > 1e-6)[0]:
    print(f"WARNING: {names[i]} has {100*lost[i]:.2f}% of its N_phi outside mix coverage (uncoverable)")
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
cons = [cp.sum(cp.multiply(F_t[t]**2, inv_nb)) <= (eps**2) * SCALE for t in range(T)]
prob = cp.Problem(cp.Minimize(cp.sum(N)), cons)    # the problem = minimize total N under uncertainty constraint
prob.solve(solver=cp.SCS, max_iters=100000)
events = N.value * SCALE                                    # back to real counts

print("status:", prob.status)
for lab, n in zip(labels, events):
    print(f"  {str(lab):18s} {n:>14,.0f}")
print(f"  {'TOTAL':18s} {events.sum():>14,.0f}   ({len(F_t)} targets, eps={eps})")

# stacked mix coverage with the optimized counts; envelope should be ~flat over target range
stack = events[:, None] * F_s                  # (S, B): events per N_phi bin from each sample
plt.stackplot(centers, stack, labels=[str(l) for l in labels])
plt.plot(centers, stack.sum(0), "k-", lw=1.2)   # the "flat line" = total n_b
plt.xlabel(r"$N_\phi$"); plt.ylabel(r"mix events per bin  $n_b=\sum_s N_s f_s$")
plt.title("Optimized mix coverage")
plt.legend(fontsize=7, ncol=2)
plt.savefig("mix_output/mix_coverage.pdf"); plt.close()

# per-target uncertainty from the optimized mix — doubles as the feasibility check
nb_opt = F_s.T @ events                              # real events per covered bin
eps_t  = np.sqrt((F_t**2 / nb_opt).sum(axis=1))      # ε_t for each target
order  = np.argsort(eps_t)
mu_t = (F_t * centers).sum(axis=1) / F_t.sum(axis=1)   # mean N_phi of each target
plt.figure()
plt.plot(mu_t, eps_t*100, "o", ms=3)
plt.axhline(eps*100, color="r", ls="--", label=f"target {eps*100:.1f}%")
plt.xlabel(r"$N_\phi$ (target mean)"); plt.ylabel(r"$\epsilon_t$ [%]")
plt.title("Per-target stat. uncertainty at optimized mix"); plt.legend()
plt.savefig("mix_output/epsilon_per_target.pdf"); plt.close()
print(f"max ε_t = {eps_t.max()*100:.3f}%  at  {names[eps_t.argmax()]}")
