import numpy as np

def reweight_weights(mix_nphi, T, m,
                     c=100.73, a=6.40, A=1.106, b=4.129,
                     bins=np.arange(200)):
    
    #Per-event reweighting factors that change the 'mix' N_phi distribution into the
    #predicted Gaussian f(T, m) -- same scheme as closure.py

    #mix_nphi : array of N_phi values, one per base ('mix') event
    #T, m     : the (T, m) point whose distribution you want to reproduce
    #c,a,A,b  : fitted f(T,m) parameters (defaults = current fit)
    #bins     : N_phi bin edges (must match how N_phi was histogrammed)

    #Returns an array of weights, same length as mix_nphi, with
    #    weight_i = Gaussian(N_phi_i; mu*, sigma*) / mix(N_phi_i)
    
    mix_nphi = np.asarray(mix_nphi)

    # f(T, m): predicted Gaussian parameters
    mu0   = c / np.sqrt(m**2 + a * T**2)        # = (mu - 0.5), the fitted quantity
    mu    = mu0 + 0.5
    x     = np.log2(T / m)
    sigma = np.sqrt(mu) * (A * 2.0**x / np.sqrt(1 + b * 2.0**(2*x)))

    # numerator = Gaussian shape, denominator = mix N_phi histogram (counts)
    centers       = bins[:-1]                           # integer-aligned (the 0.5 shift)
    mix_counts, _ = np.histogram(mix_nphi, bins=bins)
    gauss         = np.exp(-(centers - mu)**2 / (2 * sigma**2))

    # per-bin ratio (1 where the mix is empty), then look up each event's bin
    ratio = np.divide(gauss, mix_counts,
                      out=np.ones_like(gauss, dtype=float), where=mix_counts > 0)
    idx   = np.clip(np.digitize(mix_nphi, bins) - 1, 0, len(ratio) - 1)
    return ratio[idx]

w = reweight_weights(mix_nphi, T=5.0, m=5.0)   # mix_nphi = N_phi of each base event
