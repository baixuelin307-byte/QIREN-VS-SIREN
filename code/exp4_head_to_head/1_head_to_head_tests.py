# ===== Experiment 4: head-to-head tests of the best QIREN and best SIREN weights (no training) =====
# Prerequisite: run the "Restore the environment" cell of QIREN_paper_aligned_reproduction.ipynb first
# (it defines OUT, QINR, SEEDS, make_model, ImageFitting).
#
# Weights (Experiment 2, ~700 parameters each, trained on 32x32 images):
#   QIREN : OUT/best_epoch_runs/{task}__hybridren__seed{s}__best.pth
#   SIREN : OUT/baseline_best_epoch/{task}__siren__seed{s}__best.pth   (tuned SIREN)
# For each model and task we use the seed whose best-epoch weights have the lowest 32x32 error.
#
#   A. Multi-scale super-resolution : query at 32, 64, 128, 256
#   B. Frequency analysis           : reconstruction error in low / mid / high frequency bands
#   C. Weight-noise robustness       : Gaussian noise added to all parameters
#   D. Weight quantization           : 16 ... 2 bits per parameter (INR as an image codec)
import json, math, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt

for v in ('OUT', 'QINR', 'SEEDS', 'make_model', 'ImageFitting'):
    assert v in globals(), f'{v} is not defined: run the "Restore the environment" cell first'

TASKS = ('astronaut', 'camera', 'coffee')
E4 = OUT / 'exp4_head_to_head'; E4.mkdir(exist_ok=True)
torch.set_grad_enabled(False)

def psnr(mse):                       # images are in [-1, 1] (peak-to-peak 2)
    return 10 * math.log10(4 / mse)

def run(net, x):                     # train mode, as in the official code: BatchNorm uses batch statistics
    pred, _ = net(x.unsqueeze(0))
    return pred[0]

def load(model, task, seed):
    if model == 'QIREN':
        path, omega = OUT / 'best_epoch_runs' / f'{task}__hybridren__seed{seed}__best.pth', 1
        net = make_model('hybridren', 2)
    else:
        path = OUT / 'baseline_best_epoch' / f'{task}__siren__seed{seed}__best.pth'
        omega = json.loads((OUT / 'baseline_best_epoch' / f'{task}__siren__seed{seed}.json').read_text())['omega']
        net = make_model('siren', 2, omega=omega)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
    return net.train()

DATA = {(t, s): ImageFitting(s, t)[0] for t in TASKS for s in (32, 64, 128, 256)}

def mse_of(net, task, side):
    x, y = DATA[(task, side)]
    return ((run(net, x) - y) ** 2).mean().item()

# ---------- 0. Pick the best weights ----------
BEST, rows = {}, []
for model in ('QIREN', 'SIREN'):
    for task in TASKS:
        errs = {s: mse_of(load(model, task, s), task, 32) for s in SEEDS}
        s = min(errs, key=errs.get)
        BEST[(model, task)] = load(model, task, s)
        rows.append(dict(model=model, task=task, seed=s, params=sum(p.numel() for p in BEST[(model, task)].parameters()),
                         rep_mse_e3=round(errs[s] * 1e3, 3), rep_psnr=round(psnr(errs[s]), 2)))
picked = pd.DataFrame(rows)
print('===== Best weights used =====')
print(picked.to_string(index=False))

# ---------- A. Multi-scale super-resolution ----------
recA, imgs = [], {}
for (model, task), net in BEST.items():
    for side in (32, 64, 128, 256):
        x, y = DATA[(task, side)]
        p = run(net, x)
        m = ((p - y) ** 2).mean().item()
        recA.append(dict(model=model, task=task, side=side, scale=side // 32, psnr=psnr(m)))
        if side == 256: imgs[(model, task)] = p.reshape(256, 256).numpy()
A = pd.DataFrame(recA)
tabA = A.pivot_table(index=['task', 'scale'], columns='model', values='psnr').round(2)
tabA['winner'] = np.where(tabA['QIREN'] > tabA['SIREN'], 'QIREN', 'SIREN')
print('\n===== A. Multi-scale super-resolution: PSNR (dB), higher is better (scale 1 = training resolution) =====')
print(tabA.to_string())

# ---------- B. Frequency bands at the training resolution ----------
def band_errors(pred, gt, n_bands=3):
    F, G = np.fft.fftshift(np.fft.fft2(pred)), np.fft.fftshift(np.fft.fft2(gt))
    h, w = gt.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2) / (min(h, w) / 2)          # 0 = DC, 1 = Nyquist
    edges = np.linspace(0, 1.0001, n_bands + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        out.append(np.sum(np.abs(F[m] - G[m]) ** 2) / np.sum(np.abs(G[m]) ** 2))
    return out

recB = []
for (model, task), net in BEST.items():
    x, y = DATA[(task, 32)]
    e = band_errors(run(net, x).reshape(32, 32).numpy(), y.reshape(32, 32).numpy())
    for band, v in zip(('low', 'mid', 'high'), e):
        recB.append(dict(model=model, task=task, band=band, rel_error=v))
B = pd.DataFrame(recB)
tabB = B.pivot_table(index=['task', 'band'], columns='model', values='rel_error', sort=False).round(4)
tabB['winner'] = np.where(tabB['QIREN'] < tabB['SIREN'], 'QIREN', 'SIREN')
print('\n===== B. Relative spectral error per frequency band at 32x32, lower is better =====')
print(tabB.to_string())

# ---------- C. Weight-noise robustness ----------
SIGMAS, TRIALS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3), 20
recC = []
for (model, task), net in BEST.items():
    for sigma in SIGMAS:
        vals = []
        for trial in range(TRIALS if sigma > 0 else 1):
            g = torch.Generator().manual_seed(trial)
            noisy = copy.deepcopy(net)
            for p in noisy.parameters():           # noise relative to each tensor's RMS
                p.add_(sigma * p.pow(2).mean().sqrt() * torch.randn(p.shape, generator=g))
            vals.append(psnr(mse_of(noisy, task, 32)))
        recC.append(dict(model=model, task=task, sigma=sigma, psnr=float(np.mean(vals)), psnr_std=float(np.std(vals))))
C = pd.DataFrame(recC)
base = C[C.sigma == 0].set_index(['model', 'task']).psnr
C['drop'] = [b - p for b, p in zip(base.loc[list(zip(C.model, C.task))].values, C.psnr)]
tabC = C.pivot_table(index=['task', 'sigma'], columns='model', values='drop').round(2)
tabC['more robust'] = np.where(tabC['QIREN'] < tabC['SIREN'], 'QIREN', 'SIREN')
print(f'\n===== C. Weight noise: PSNR drop (dB) at 32x32, mean of {TRIALS} trials, lower is better =====')
print('sigma = noise std relative to the RMS of each parameter tensor')
print(tabC[tabC.index.get_level_values('sigma') > 0].to_string())

# ---------- D. Weight quantization (INR as an image codec) ----------
def quantize(net, bits):
    q = copy.deepcopy(net)
    for p in q.parameters():                   # symmetric uniform quantization per tensor
        m = p.abs().max()
        if m == 0: continue
        levels = 2 ** (bits - 1) - 1
        p.copy_(torch.round(p / m * levels) / levels * m)
    return q

recD = []
for (model, task), net in BEST.items():
    n = sum(p.numel() for p in net.parameters())
    for bits in (32, 16, 8, 6, 4, 3, 2):
        q = net if bits == 32 else quantize(net, bits)
        recD.append(dict(model=model, task=task, bits=bits, bpp=n * bits / (32 * 32), psnr=psnr(mse_of(q, task, 32))))
D = pd.DataFrame(recD)
tabD = D.pivot_table(index=['task', 'bits'], columns='model', values='psnr').round(2).sort_index(ascending=[True, False])
tabD['winner'] = np.where(tabD['QIREN'] > tabD['SIREN'], 'QIREN', 'SIREN')
print('\n===== D. Weight quantization: PSNR (dB) at 32x32, higher is better =====')
print('bits per pixel = parameters x bits / 1024 pixels (QIREN 657 and SIREN 701 parameters)')
print(tabD.to_string())

# ---------- Overall scoreboard ----------
def wins(tab, col):
    w = tab[col].value_counts()
    return f"QIREN {w.get('QIREN', 0)} : {w.get('SIREN', 0)} SIREN"
board = pd.DataFrame([
    ('A. Super-resolution, scale 1 (training resolution)', wins(tabA[tabA.index.get_level_values('scale') == 1], 'winner')),
    ('A. Super-resolution, scales 2-8 (unseen resolutions)', wins(tabA[tabA.index.get_level_values('scale') > 1], 'winner')),
    ('B. Low-frequency error', wins(tabB[tabB.index.get_level_values('band') == 'low'], 'winner')),
    ('B. Mid-frequency error', wins(tabB[tabB.index.get_level_values('band') == 'mid'], 'winner')),
    ('B. High-frequency error', wins(tabB[tabB.index.get_level_values('band') == 'high'], 'winner')),
    ('C. Robustness to weight noise', wins(tabC[tabC.index.get_level_values('sigma') > 0], 'more robust')),
    ('D. Quality at 8-16 bits', wins(tabD[tabD.index.get_level_values('bits').isin([8, 16])], 'winner')),
    ('D. Quality at 2-6 bits', wins(tabD[tabD.index.get_level_values('bits').isin([2, 3, 4, 6])], 'winner')),
], columns=['test', 'wins'])
print('\n===== Scoreboard (number of task/setting cases won) =====')
print(board.to_string(index=False))

# ---------- Save tables and figures ----------
for name, df in (('picked_weights', picked), ('A_super_resolution', A), ('B_frequency_bands', B),
                 ('C_weight_noise', C), ('D_quantization', D), ('scoreboard', board)):
    df.to_csv(E4 / f'{name}.csv', index=False)

col = {'QIREN': 'tab:red', 'SIREN': 'tab:blue'}
fig, ax = plt.subplots(4, 3, figsize=(14, 15))
for j, task in enumerate(TASKS):
    for model in ('QIREN', 'SIREN'):
        a = A[(A.model == model) & (A.task == task)]
        ax[0, j].plot(a.scale, a.psnr, 'o-', c=col[model], label=model)
        b = B[(B.model == model) & (B.task == task)]
        ax[1, j].bar(np.arange(3) + (0.2 if model == 'SIREN' else -0.2), b.rel_error, 0.4, color=col[model], label=model)
        c = C[(C.model == model) & (C.task == task) & (C.sigma > 0)]
        ax[2, j].errorbar(c.sigma, c['drop'], yerr=c.psnr_std, fmt='o-', c=col[model], label=model, capsize=3)
        d = D[(D.model == model) & (D.task == task)]
        ax[3, j].plot(d.bpp, d.psnr, 'o-', c=col[model], label=model)
    ax[0, j].set(title=f'{task}: super-resolution', xlabel='scale (x training resolution)', ylabel='PSNR (dB)', xscale='log', xticks=[1, 2, 4, 8], xticklabels=['1', '2', '4', '8'])
    ax[1, j].set(title=f'{task}: spectral error at 32x32', xticks=range(3), xticklabels=['low', 'mid', 'high'], ylabel='relative error', yscale='log')
    ax[2, j].set(title=f'{task}: weight noise', xlabel='relative noise sigma', ylabel='PSNR drop (dB)', xscale='log')
    ax[3, j].set(title=f'{task}: quantization', xlabel='bits per pixel', ylabel='PSNR (dB)', xscale='log')
    for r in range(4):
        ax[r, j].legend(); ax[r, j].grid(alpha=0.3)
        ax[r, j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.tight_layout(); plt.savefig(E4 / 'exp4_curves.png', dpi=110); plt.close()

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for j, task in enumerate(TASKS):
    ax[j, 0].imshow(DATA[(task, 256)][1].reshape(256, 256).numpy(), cmap='gray', vmin=-1, vmax=1); ax[j, 0].set_title(f'{task}: ground truth 256x256')
    for k, model in enumerate(('QIREN', 'SIREN'), 1):
        pv = A[(A.model == model) & (A.task == task) & (A.side == 256)].psnr.item()
        ax[j, k].imshow(imgs[(model, task)], cmap='gray', vmin=-1, vmax=1); ax[j, k].set_title(f'{model} (x8): {pv:.2f} dB')
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.savefig(E4 / 'exp4_super_resolution_x8.png', dpi=110); plt.close()

print('\nSaved tables and figures to', E4)
try:
    from IPython.display import Image as _Img, display
    display(_Img(str(E4 / 'exp4_curves.png'))); display(_Img(str(E4 / 'exp4_super_resolution_x8.png')))
except Exception:
    pass
