# -*- coding: utf-8 -*-
"""
ESN + Diffusion Maps (1 tangent + 1 normal) with a 3-chart atlas and
a UNIFORM SEC certificate over the evaluated tube.

No leak, no biases anywhere.
Closed-loop SEC: encoder/decoder are implicitly accounted for by fitting
the reduced map directly on (s,y) → (s',y').

Outputs:
- Per-time SEC1/SEC2 (diagnostics)
- Uniform region certificate using inf/sup bounds across the tube:
    C_inf, My_sup, Ly_sup, Ms_sup
  and the polynomial / margins.

Author: (you)
"""

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

plt.close('all')

# =========================
# CONFIG
# =========================
SEED    = 7
N       = 200
RHO     = 0.56  #0.56
RIDGE   = 5e-2            # ESN readout ridge (two-output linear readout)
WASH    = 2000
N_TRAIN = 10000
N_TEST  = 5000
PLOT_FREE_STEPS = N_TEST

# Encoding
AMP     = 1.0
TWOPI   = 2.0*np.pi

# W_in init & calibration (gain only)
W_IN_INIT_SCALE = 0.3
GPRIME_TARGET   = 0.55
SAT_TARGET      = 0.05
MAX_W_IN_GAIN   = 2.0

# High-precision doubling map
DECIMAL_PREC = 200
X0_DECIMAL   = None

# Jacobian/atan2 smoothing
ATAN2_DELTA  = 1e-6

# ===== Diffusion maps / reduced chart =====
EVAL_LAST       = N_TRAIN - WASH # 2000     # tail length to build reduced chart/SEC
DM_EPS_SCALE    = 1.0
DM_K            = 3        # compute >=2 for (φ1, φ2)

# Tangent s options
USE_ARC_LENGTH_S = True
AUTO_ROTATE_SEAM = True
SEAM_EPS         = 0.01

# Normal coordinate(s)
RFOURIER_K       = 10
RFOURIER_RIDGE   = 3e-4

# Local linear regression (A_t)
KNN_Z        = 200         # neighbors; we split ≈ half/half by s-side
RIDGE_A      = 1e-4        # base ridge
OUTLIER_STD  = 4.0         # (kept for completeness; not used in balanced fit)
SMOOTH_A_WIN = 7           # odd; 1 disables

# Robustness knobs
MIN_SPREAD   = 1e-5        # min std for design columns before whitening
MAX_COND     = 1e8         # condition number threshold (soft)
RIDGE_BUMP   = 50.0        # ridge multiplier if ill-conditioned
HUBER_DELTA  = 1.0         # Huber parameter for residual reweight
EPS_SCORE    = 1e-12       # to avoid div-by-zero in score

# SEC tolerances (purely numerical)
TOL_DISC = 1e-12
TOL_POS  = 1e-10

# =========================
# Helpers
# =========================
def med(v): 
    return float(np.nanmedian(v))

def qtls(v, qs=[0.1,0.5,0.9,0.99]): 
    return np.quantile(v, qs)

def circ_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)

def wrap01(x):  # map to [0,1)
    return (x % 1.0 + 1.0) % 1.0

def wrap_diff_s(ds):  # map a difference to (-0.5,0.5]
    return (ds + 0.5) % 1.0 - 0.5

# ========== Doubling map ==========
getcontext().prec = DECIMAL_PREC
def doubling_series_decimal(n, x0=None):
    if x0 is None:
        rt2 = Decimal(2).sqrt() 
        x = rt2 - int(rt2)
    else:
        x = Decimal(str(x0))
        x = x - int(x)
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        out[t] = float(x) % 1.0
        x = (x * 2) % 1
    return out

# ========== Encoding ==========
def enc(y_scalar):
    th = TWOPI * y_scalar
    return np.array([AMP*np.sin(th), AMP*np.cos(th)], dtype=np.float64)

def du_dy(y_scalar):
    th = TWOPI * y_scalar
    return np.array([AMP*TWOPI*np.cos(th), -AMP*TWOPI*np.sin(th)], dtype=np.float64)

# ========== Reservoir utils ==========
#check what this does and if changing it to some other W helps ?
def orthonormal_matrix(n, rho):
    W = np.zeros((n, n))
    W[0, n-1] = 1
    for i in range(n-1):
        W[i+1, i] = 1
    
    W*= rho/np.max(np.abs(np.linalg.eigvals(W))) 
    return W

def normalize_rows(V, eps=1e-12):
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return V / nrm

def decode_to_y(v2):
    s, c = v2[...,0], v2[...,1]
    ang = np.arctan2(s, c)
    return (ang % (2*np.pi)) / (2*np.pi)
'''
def spec_norm(A, iters=6, seed=0):
    A = np.asarray(A, np.float64)
    if A.size == 0: 
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1]); v /= np.linalg.norm(v) + 1e-300
    for _ in range(iters):
        w = A @ v; s = np.linalg.norm(w)
        if s == 0.0: return 0.0
        v = A.T @ (w / (s + 1e-300))
        v /= np.linalg.norm(v) + 1e-300
    return float(np.linalg.norm(A @ v))
'''
# ========== Full closed-loop Jacobian (info) ==========
def jacobians_closed_loop_circle(W, W_in, W_out, H_seq, Y_seq, Z_seq, delta=1e-6):
    N = W.shape[0] 
    T = len(Y_seq)
    ws = np.asarray(W_out[:,0], np.float64)
    wc = np.asarray(W_out[:,1], np.float64)
    W  = np.asarray(W, np.float64)
    Wi = np.asarray(W_in, np.float64)
    J  = np.empty((T, N, N), dtype=np.float64)
    for t in range(T):
        h = np.asarray(H_seq[t], np.float64)
        z = np.asarray(Z_seq[t], np.float64)
        s = float(h @ ws) 
        c = float(h @ wc)
        denom = s*s + c*c + float(delta)
        dy_dh = (c*ws - s*wc)/(2*np.pi*denom)
        #b = Wi @ du_dy(Y_seq[t])
        v2 = h@W_out
        yhat_t = decode_to_y(v2)
        b = Wi @du_dy(yhat_t)
        
        D = 1.0/(np.cosh(z)**2)
        Wstar = W + np.outer(b, dy_dh)
        J[t] = D[:,None] * Wstar
    return J

def top_lyapunov(J):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(J.shape[1]);
    #v /= np.linalg.norm(v) + 1e-300
    ssum = 0.0
    for t in range(J.shape[0]):
        v = J[t] @ v
        n = np.linalg.norm(v) 
        ssum += np.log(n + 1e-300)
        v /= n + 1e-300      ###< ------------------------------------------------------ This might need fixing
    return ssum / J.shape[0]

# ========== ESN build & data ==========
np.random.seed(SEED)
W =  orthonormal_matrix(N, RHO)
W_in = (np.random.randn(N, 2) / np.sqrt(2)) * W_IN_INIT_SCALE

TOTAL = WASH + N_TRAIN + N_TEST + 1
x = doubling_series_decimal(TOTAL, x0=X0_DECIMAL)
U = np.stack([enc(u) for u in x[:-1]], axis=0)
Y = x[1:]
Y2 = np.column_stack([np.sin(TWOPI*Y), np.cos(TWOPI*Y)])

# Calibrate W_in (scale only, cap)
def run_washout_stats(U, W, W_in, num=WASH):
    h = np.zeros(N, dtype=np.float64)
    gprime_vals, sat_flags = [], []
    for t in range(num):
        z = W @ h + W_in @ U[t]
        gprime = 1.0 / np.cosh(z)**2
        gprime_vals.append(np.mean(gprime))
        sat_flags.append(np.mean(np.abs(z) > 2.0))
        h = np.tanh(z)
    return float(np.mean(gprime_vals)), float(np.mean(sat_flags))

def calibrate_W_in(U, W, W_in, target_gprime=GPRIME_TARGET, target_sat=SAT_TARGET, max_iter=10, max_gain=MAX_W_IN_GAIN):
    scale = 1.0
    for _ in range(max_iter):
        gp, fs = run_washout_stats(U, W, W_in * scale)
        adj = 1.0
        if fs > target_sat:
            adj *= 0.7
        if gp < target_gprime: 
            adj *= 0.85
        elif gp > target_gprime+0.05: 
            adj *= 1.15
        new_scale = min(scale * adj, max_gain)
        if abs(new_scale - scale) < 1e-3:
            scale = new_scale; break
        scale = new_scale
    return W_in * scale, scale

W_in, scale_used = calibrate_W_in(U, W, W_in)
print(f"[Auto-calibration] (capped) W_in gain = {scale_used:.3f}")

# Teacher forcing pass
def run_reservoir(U):
    T = U.shape[0]
    H = np.zeros((T, N), dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    gprime_mean = 0.0
    frac_sat = 0.0
    for t in range(T):
        z = W @ h + W_in @ U[t]
        gprime = 1.0 / np.cosh(z)**2
        gprime_mean += np.mean(gprime)
        frac_sat    += np.mean(np.abs(z) > 2.0)
        h = np.tanh(z)
        H[t] = h
    return H, gprime_mean/T, frac_sat/T

H_all, gprime_mean, frac_sat = run_reservoir(U)
print(f"[Saturation] mean g'(z)={gprime_mean:.3f},  frac(|z|>2)={frac_sat:.3f}")

# Train readout (no bias)
idx0 = WASH
idx1 = WASH + N_TRAIN
H_tr  = H_all[idx0:idx1]
Y2_tr = Y2[idx0:idx1]
A = H_tr.T @ H_tr + RIDGE * np.eye(N)
B = H_tr.T @ Y2_tr
W_out = np.linalg.solve(A, B)

# Teacher test
H_te  = H_all[idx1:idx1+N_TEST]
Y_te  = Y[idx1:idx1+N_TEST]
V2_te = normalize_rows(H_te @ W_out)
Yhat_te = decode_to_y(V2_te)
err_te = circ_dist(Y_te, Yhat_te)
print(f"[Teacher forcing]  MSE_circ = {np.mean(err_te**2):.6e},  MAE_circ = {np.mean(err_te):.6e}")

# Closed-loop rollout
T_free = N_TEST
Y_free = np.zeros(T_free)
H_free = np.zeros((T_free + 1, N))
Z_free = np.zeros((T_free, N))
H_free[0] = H_all[idx1 - 1].copy()
u_vec = U[idx1].copy()
for t in range(T_free):
    z = W @ H_free[t] + W_in @ u_vec
    Z_free[t] = z
    h_next = np.tanh(z)
    H_free[t+1] = h_next
    v2 = h_next @ W_out
    v2n = v2 #/ (np.linalg.norm(v2) + 1e-12)  ##########<----- deviding by ||v2 + \epsilon||
    yhat = decode_to_y(v2n)
    Y_free[t] = yhat
    u_vec = enc(yhat)

def report_readout_radius(H_seq, W_out, tag="autonomous"):
    V = H_seq @ W_out
    r = np.linalg.norm(V, axis=1)
    q = qtls(r)
    print(f"[{tag} radius] r quantiles: 10%={q[0]:.3f}, 50%={q[1]:.3f}, 90%={q[2]:.3f}, 99%={q[3]:.3f}")
    return r
_ = report_readout_radius(H_te,  W_out, tag="teacher")
_ = report_readout_radius(H_free[1:], W_out, tag="autonomous")

# Full Jacobians & Lyapunov (info)
J_full = jacobians_closed_loop_circle(W, W_in, W_out, H_free, Y_free, Z_free, delta=ATAN2_DELTA)
lam = top_lyapunov(J_full)
print(f"\n[Lyapunov] top lambda approx {lam:.4f}   (log 2 approx {np.log(2.0):.4f})")

# =========================
# Diffusion Maps on the tail
# =========================
tail = min(EVAL_LAST, T_free-1)   # need t and t+1 pairs
t0   = T_free - tail
Xtail = H_free[1:][t0: t0+tail]   # states h_{t+1} on tail

# pairwise squared distances (Gram trick)
S = np.sum(Xtail*Xtail, axis=1)
G = Xtail @ Xtail.T
D2 = S[:,None] + S[None,:] - 2.0*G
D2 = np.maximum(D2, 0.0)

mask = ~np.eye(tail, dtype=bool)
eps = DM_EPS_SCALE * np.median(D2[mask])
eps = max(eps, 1e-12)
K = np.exp(-D2 / eps)
d = np.sum(K, axis=1)
d_sqrt = np.sqrt(np.maximum(d, 1e-300))
A_sym = (K / d_sqrt[:,None]) / d_sqrt[None,:]

# eigendecomposition
evals, Ueig = np.linalg.eigh(A_sym)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
Ueig  = Ueig[:, idx]

# diffusion coordinates: \psi = D^{-1/2} U, then \phi = \\lambda \psi  < ------------------------------change here
DM_K_eff = max(1, min(DM_K, Ueig.shape[1]-1))
U_nontriv = Ueig[:, 1:1+DM_K_eff]
lam_nontriv = evals[1:1+DM_K_eff]
psi = (U_nontriv / d_sqrt[:,None])
phi = psi * lam_nontriv[None,:]              # (tail, DM_K_eff)
phi1 = phi[:,0]
phi2 = phi[:,1] if phi.shape[1] >= 2 else np.zeros_like(phi1)

# =========================
# Chart builder
# =========================
def fourier_design(z, K):
    z = np.asarray(z).reshape(-1)
    T = z.shape[0]
    Phi = np.zeros((T, 2*K+1), dtype=np.float64)
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        ang = 2*np.pi*k*z
        Phi[:,2*k-1] = np.cos(ang)
        Phi[:,2*k]   = np.sin(ang)
    return Phi

def build_chart(phi1, phi2, offset=0.0, use_arc=True):
    # base angle
    theta = np.arctan2(phi2, phi1)
    s_angle = wrap01(theta / (2*np.pi))
    # arc-length parameterization
    if use_arc:
        P = np.column_stack([phi1, phi2])
        ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s_arc = np.concatenate([[0.0], np.cumsum(ds)])
        s_arc = (s_arc - s_arc.min()) / max(1e-12, (s_arc.max() - s_arc.min()))
        s_base = s_arc
    else:
        s_base = s_angle
    s_tail = (s_base - offset) % 1.0

    # radial normal
    r_tail = np.sqrt(phi1**2 + phi2**2)
    Phi_s  = fourier_design(s_tail, RFOURIER_K)
    A_r    = np.linalg.solve(Phi_s.T @ Phi_s + RFOURIER_RIDGE*np.eye(Phi_s.shape[1]),
                             Phi_s.T @ r_tail)
    r_hat  = Phi_s @ A_r
    y1_tail= r_tail - r_hat
    y_tail = y1_tail[:,None]

    # scales
    s_step = np.median(np.abs(wrap_diff_s(np.diff(s_tail))))
    s_scale= max(s_step, 1e-3)
    y_std_vec = np.std(y_tail, axis=0)
    y_std_vec[y_std_vec < 1e-9] = 1e-9

    # pairs
    z_all      = np.column_stack([s_tail, y_tail])          # (tail, 2)
    z_next_all = np.roll(z_all, -1, axis=0)
    Tt         = z_all.shape[0] - 1
    z          = z_all[:Tt]
    z_next     = z_next_all[:Tt]

    return s_tail, y_tail, z, z_next, s_scale, y_std_vec

# auto seam offset (chart 0)
if AUTO_ROTATE_SEAM:
    grid = np.linspace(0, 1, 360, endpoint=False)
    theta0 = np.arctan2(phi2, phi1)
    if USE_ARC_LENGTH_S:
        P0 = np.column_stack([phi1, phi2])
        ds0 = np.linalg.norm(np.diff(P0, axis=0), axis=1)
        s_arc0 = np.concatenate([[0.0], np.cumsum(ds0)])
        s_arc0 = (s_arc0 - s_arc0.min()) / max(1e-12, (s_arc0.max() - s_arc0.min()))
        s_base0 = s_arc0
    else:
        s_base0 = wrap01(theta0 / (2*np.pi))
    best_off, best_score = 0.0, 1e18
    for off in grid:
        s_shift = (s_base0 - off) % 1.0
        score = np.sum((s_shift < SEAM_EPS) | (s_shift > 1.0 - SEAM_EPS))
        if score < best_score:
            best_score, best_off = score, off
    off0 = best_off
else:
    off0 = 0.0

# 3-chart atlas: seams at 0°, 120°, 240° relative to off0
off_list = np.mod(off0 + np.array([0.0, 1/3, 2/3]), 1.0)

charts = [build_chart(phi1, phi2, offset=o, use_arc=USE_ARC_LENGTH_S) for o in off_list]
Tt = charts[0][2].shape[0]
assert all(c[2].shape[0] == Tt for c in charts)

# =========================
# Local regression (balanced L/R, robust, conditioned)
# =========================
def z_distance(z0, z1, s_scale, y_std_vec):
    ds = np.abs(wrap_diff_s(z0[0] - z1[0])) / s_scale
    dy = (z0[1:] - z1[1:]) / y_std_vec
    return np.sqrt(ds*ds + np.dot(dy, dy))

def huber_weights(r, delta=1.5):
    a = np.abs(r) <= delta
    w = np.empty_like(r)
    w[a]  = 1.0
    w[~a] = (delta / (np.abs(r[~a]) + 1e-12))
    return w

def local_lin_jacobian_balanced(z_all, z_next_all, t, k_nn=KNN_Z, ridge=RIDGE_A,
                                s_scale=1.0, y_std_vec=None):
    Tloc, d = z_all.shape
    z_t  = z_all[t]
    z_tp = z_next_all[t]

    # distances & sort
    dists = np.array([z_distance(z_t, z_all[i], s_scale, y_std_vec) for i in range(Tloc)])
    order = np.argsort(dists)

    # split by s-side for balance
    ds_all = wrap_diff_s(z_all[:,0] - z_t[0])
    left  = order[ds_all[order] < 0]
    right = order[ds_all[order] >= 0]

    half = max((k_nn)//2, 3*d+2)  # ensure enough per side
    idx_left  = left[:half]
    idx_right = right[:half]
    idx = np.unique(np.concatenate([idx_left, idx_right]))
    # if still not enough, top up globally
    need = max(2*half, 3*d+2)
    if idx.size < need:
        idx = np.unique(np.concatenate([idx, order[:need]]))

    # center with seam-aware ds
    s_i      = z_all[idx, 0];   y_i   = z_all[idx, 1]
    s_ip1    = z_next_all[idx,0]; y_ip1 = z_next_all[idx,1]

    ds_local      = wrap_diff_s(s_i   - z_t[0])
    ds_next_local = wrap_diff_s(s_ip1 - z_tp[0])

    X = np.column_stack([ds_local,      y_i   - z_t[1]])
    Y = np.column_stack([ds_next_local, y_ip1 - z_tp[1]])

    # standardize columns
    col_std_X = np.std(X, axis=0); col_std_X[col_std_X < MIN_SPREAD] = MIN_SPREAD
    col_std_Y = np.std(Y, axis=0); col_std_Y[col_std_Y < MIN_SPREAD] = MIN_SPREAD
    Xn = X / col_std_X[None,:]
    Yn = Y / col_std_Y[None,:]

    # Gaussian distance weights
    sig = np.median(dists[idx]); sig = max(sig, 1e-6)
    w0 = np.exp(-0.5 * (dists[idx]/sig)**2)

    # First pass
    Xw = Xn * w0[:,None]; Yw = Yn * w0[:,None]
    XtWX = Xw.T @ Xn + ridge*np.eye(d)
    XtWY = Xw.T @ Yn
    cond0 = np.linalg.cond(XtWX)
    if cond0 > MAX_COND:
        XtWX = Xw.T @ Xn + (RIDGE_BUMP*ridge)*np.eye(d)
    An = np.linalg.solve(XtWX, XtWY).T       # (2x2) normalized
    # residuals & Huber reweight
    Rn = Yn - Xn @ An.T
    rnorm = np.linalg.norm(Rn, axis=1)
    wr = huber_weights(rnorm, delta=HUBER_DELTA)
    w  = w0 * wr
    # Second pass
    Xw2 = Xn * w[:,None]; Yw2 = Yn * w[:,None]
    XtWX2 = Xw2.T @ Xn + ridge*np.eye(d)
    XtWY2 = Xw2.T @ Yn
    cond1 = np.linalg.cond(XtWX2)
    if cond1 > MAX_COND:
        XtWX2 = Xw2.T @ Xn + (RIDGE_BUMP*ridge)*np.eye(d)
    An2 = np.linalg.solve(XtWX2, XtWY2).T
    A  = (An2 * (col_std_Y[:,None])) / (col_std_X[None,:])
    return A, cond1

def fit_A_series(z, z_next, s_scale, y_std):
    Tt = z.shape[0]
    A_list = []
    cond_list = []
    for t in range(Tt):
        A, c = local_lin_jacobian_balanced(z, z_next, t,
                                           k_nn=KNN_Z, ridge=RIDGE_A,
                                           s_scale=s_scale, y_std_vec=y_std)
        A_list.append(A); cond_list.append(c)
    A_arr = np.stack(A_list, axis=0)
    cond_arr = np.array(cond_list, dtype=np.float64)
    # temporal smoothing
    if SMOOTH_A_WIN > 1 and SMOOTH_A_WIN % 2 == 1:
        half = SMOOTH_A_WIN // 2
        w = np.hanning(SMOOTH_A_WIN); w /= w.sum()
        A_s = A_arr.copy()
        for i in range(half, Tt-half):
            A_s[i] = np.tensordot(w, A_arr[i-half:i+half+1], axes=(0,0))
        A_arr = A_s
    return A_arr, cond_arr

# fit each chart
A_list = []
cond_list = []
seam_dists = []
for (s_tail, y_tail, z, z_next, s_scale, y_std_vec) in charts:
    A_arr, cond_arr = fit_A_series(z, z_next, s_scale, y_std_vec)
    A_list.append(A_arr)
    cond_list.append(cond_arr)
    seam_dists.append(np.minimum(s_tail[:-1], 1.0 - s_tail[:-1]))  # per-time seam distance

A_list  = np.stack(A_list, axis=0)   # (C, Tt, 2, 2)
cond_all= np.stack(cond_list, axis=0)# (C, Tt)
dseam   = np.stack(seam_dists, axis=0)# (C, Tt)

# condition-number–aware chart selection
# score = seam_distance / (1 + log10(cond+1))
score = dseam / (1.0 + np.log10(cond_all + 1.0 + EPS_SCORE))
best_idx = np.argmax(score, axis=0)             # (Tt,)
rows = np.arange(A_list.shape[1])
A_arr = A_list[best_idx, rows]                  # (Tt, 2, 2)

# =========================
# SEC constants (2x2 split)
# =========================
C_red   = np.abs(A_arr[:,0,0])
Ly_red = np.abs(A_arr[:,0,1])
Ms_red = np.abs(A_arr[:,1,0])
My_red   = np.abs(A_arr[:,1,1])

def sec_flags(C, M, Ly, Ms, tol_disc=TOL_DISC, tol_pos=TOL_POS):
    disc = (C - M)**2 - 4.0*(Ly*Ms)
    disc_eff = max(disc, -tol_disc)
    sd = np.sqrt(max(disc_eff, 0.0))
    r1 = 0.5*((C - M) - sd)
    r2 = 0.5*((C - M) + sd)
    pos = (r1 > -tol_pos) and (r2 > -tol_pos)
    sec1 = pos
    xi_min = min(r1, r2)
    sec2 = pos and ((xi_min + M) < 1.0 + tol_pos)
    return sec1, sec2, disc, xi_min

# Per-time SEC (diagnostics)
Tt = C_red.shape[0]
SEC1 = np.zeros(Tt, dtype=bool)
SEC2 = np.zeros(Tt, dtype=bool)
disc_vec = np.zeros(Tt)
xi_min_vec = np.zeros(Tt)
for t in range(Tt):
    s1b, s2b, dsc, xim = sec_flags(C_red[t], My_red[t], Ly_red[t], Ms_red[t])
    SEC1[t], SEC2[t] = s1b, s2b
    disc_vec[t]      = dsc
    xi_min_vec[t]    = xim

# =========================
# UNIFORM certificate on the tube
# =========================
C_inf    = float(np.min(C_red))
My_sup    = float(np.max(My_red))
Ly_sup  = float(np.max(Ly_red))
Ms_sup  = float(np.max(Ms_red))

disc_u = (C_inf - My_sup)**2 - 4.0*(Ly_sup*Ms_sup)
sd_u   = np.sqrt(max(disc_u, 0.0))
r1_u   = 0.5*((C_inf - My_sup) - sd_u)
r2_u   = 0.5*((C_inf - My_sup) + sd_u)
sec1_u = (disc_u >= -TOL_DISC) and (r1_u > -TOL_POS) and (r2_u > -TOL_POS)
xi_min_u = min(r1_u, r2_u)
sec2_u = sec1_u and ((xi_min_u + My_sup) < 1.0 + TOL_POS)

# margins
disc_margin   = disc_u
sec2_margin   = 1.0 - (xi_min_u + My_sup)
gap_CM_margin = (C_inf - My_sup)

# =========================
# Reports
# =========================
print("\n=== Diffusion Maps / Reduced SEC (3-chart atlas) on last {} steps ===".format(tail))
print("Eigenvalues (nontrivial) λ1..:", evals[1:1+min(5, DM_K_eff)])
for i, o in enumerate(off_list):
    s_tail = charts[i][0]
    y_tail = charts[i][1]
    s_step = np.median(np.abs(wrap_diff_s(np.diff(s_tail))))
    print(f"[Chart {i}] seam offset={o:.3f} | s-step median={s_step:.6g} | y std={float(np.std(y_tail)):.3g}")

print("\n[Reduced SEC stats (per-time)]")
print(" median C     =", med(C_red))
print(" median My     =", med(My_red))
print(" median L_y   =", med(Ly_red))
print(" median M_s   =", med(Ms_red))
sec1_hits = int(np.count_nonzero(SEC1))
sec2_hits = int(np.count_nonzero(SEC2))
print(f" SEC1 per-time: {sec1_hits}/{Tt} = {sec1_hits/Tt:.4f} → {100.0*sec1_hits/Tt:.2f}%")
print(f" SEC2 per-time: {sec2_hits}/{Tt} = {sec2_hits/Tt:.4f} → {100.0*sec2_hits/Tt:.2f}%")

print("\n=== UNIFORM SEC certificate on this tube ===")
print(f" C_inf    = {C_inf:.9f}")
print(f" M_y_sup    = {My_sup:.9f}")
print(f" L_y_sup  = {Ly_sup:.9f}")
print(f" M_s_sup  = {Ms_sup:.9f}")
print(f" disc_u   = {disc_u:.9e}")
print(f" roots_u  = ({r1_u:.9f}, {r2_u:.9f})")
print(f" xi_min_u + M_y_sup = {xi_min_u + My_sup:.9f}   (SEC2 needs < 1)")
print(f" SEC1(uniform) = {sec1_u}")
print(f" SEC2(uniform) = {sec2_u}")
print(f" margins:  disc={disc_margin:.3e},  SEC2={sec2_margin:.3e},  (C_inf - M_y_sup)={gap_CM_margin:.3e}")

# =========================
# Plots
# =========================
K = min(600, N_TEST)
Mpl = min(PLOT_FREE_STEPS, N_TEST)

plt.figure(figsize=(11,7))
plt.subplot(3,1,1)
plt.title("Teacher-forced one-step (first {} test steps)".format(K))
plt.plot(Y_te[:K], lw=1.0, label="truth y")
plt.plot(Yhat_te[:K], lw=1.0, label="ESN 1-step (decoded)")
plt.ylabel("y"); plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)

plt.subplot(3,1,2)
plt.title("Autonomous rollout (decoded) vs truth segment")
plt.plot(Y_te[:Mpl], lw=1.0, label="truth segment")
plt.plot(Y_free[:Mpl], lw=1.0, label="ESN free-run (decoded)")
plt.ylabel("y"); plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)

plt.subplot(3,1,3)
plt.title("Diffusion map embedding (tail) and 3 seams")
plt.plot(phi1, phi2, '.', ms=2, alpha=0.7, label="tail")
r0 = np.sqrt(phi1**2 + phi2**2).mean()
for i, o in enumerate(off_list):
    ang = 2*np.pi*o
    plt.plot([0, r0*np.cos(ang)], [0, r0*np.sin(ang)], lw=2, label=f"seam {i}")
plt.xlabel(r'$\phi_1$'); plt.ylabel(r'$\phi_2$'); plt.axis('equal'); plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Plot the last subplot on its own
# Standalone figure + axis
fig, ax = plt.subplots(figsize=(5, 4.2))

ax.set_title("Diffusion map embedding (tail) and 3 seams")

# Scatter of tail
ax.plot(phi1, phi2, '.', ms=2, alpha=0.7, label="tail")

# Seams
r0 = np.sqrt(phi1**2 + phi2**2).mean()
for i, o in enumerate(off_list):
    ang = 2*np.pi*o
    ax.plot([0, r0*np.cos(ang)], [0, r0*np.sin(ang)], lw=2, label=f"seam {i}")

# LaTeX-style labels (mathtext)
ax.set_xlabel(r'$\phi_1$')
ax.set_ylabel(r'$\phi_2$')

# Nice looking axes
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Save (vector PDF recommended for LaTeX)
fig.savefig("dmap_tail_seams.pdf", bbox_inches="tight")

# Optional: visualize reduced SEC signals on a short window
try:
    WN = min(600, Tt)
    sl = slice(-WN, None)
    plt.figure(figsize=(10,4.6))
    plt.plot(C_red[sl], label="C (=|a11|)")
    plt.plot(My_red[sl], label="My (=|a22|)")
    plt.plot(Ly_red[sl], label="Ly (=|a01|)")
    plt.plot(Ms_red[sl], label="Ms (=|a10|)")
    plt.title("Reduced SEC constants (tail, atlas-selected)")
    plt.xlabel("time (tail index)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,4.2))
    plt.plot(SEC1.astype(float)[sl], label="SEC1 (per-time)")
    plt.plot(SEC2.astype(float)[sl], label="SEC2 (per-time)")
    plt.title("SEC1/SEC2 flags (1=true) in reduced chart (atlas-selected)")
    plt.xlabel("time (tail index)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
except Exception as e:
    print("SEC plots skipped:", e)
