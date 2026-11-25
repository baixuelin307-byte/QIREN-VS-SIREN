# ===== QIREN on SIREN's video task (Cat): fast exact GPU simulation; resumable; saves the best weights =====
import os, sys, time, json, copy, subprocess, importlib.util
from pathlib import Path

# ---------- 1. Drive ----------
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/QIREN_paper_recheck'); OUT.mkdir(parents=True, exist_ok=True)

# ---------- 2. Official QIREN code (pinned commit) + required patches ----------
ROOT = Path('/content/QIREN_paper_recheck_repo'); QINR = ROOT / 'qinr'
SHA = '56ab4010ee3fbb3371c9d5d5f5284a8c41d4e6f3'
if not QINR.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/GGorMM1/QIREN.git', str(ROOT)], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'checkout', '-q', SHA], check=True)
mp = QINR / 'modules.py'
s = mp.read_text(encoding='utf-8')
s = s.replace('import qiskit.providers.aer.noise as noise', '# import qiskit.providers.aer.noise as noise  # unused')
s = s.replace('qml.RZ(inputs[j], wires=j)', 'qml.RZ(inputs[..., j], wires=j)')
mp.write_text(s, encoding='utf-8')
try:
    import pennylane
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pennylane'], check=True)
spec = importlib.util.spec_from_file_location('qiren_modules', mp)
qiren_modules = importlib.util.module_from_spec(spec); spec.loader.exec_module(qiren_modules)

import numpy as np, torch
from torch.utils.data import DataLoader
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU')

def make_qiren(in_f=2, out_f=1):
    return qiren_modules.Hybridren(in_features=in_f, out_features=out_f, hidden_features=8,
                                   hidden_layers=2, spectrum_layer=2, use_noise=0)

# ---------- 3. Exact fast implementation of the quantum layer (mathematically identical to the official PennyLane circuit) ----------
#   Official circuit: |psi> = SEL(w2) · D(x) · SEL(w1[1]) · D(x) · SEL(w1[0]) |0>, output <Z_i>
#   SEL depends only on the weights and D(x) = RZ(x_0)⊗...⊗RZ(x_7) is a diagonal phase, so the circuit is a sequence of matrix products
def _rot(w):                     # qml.Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
    phi, th, om = w[..., 0], w[..., 1], w[..., 2]
    c, sn = torch.cos(th / 2), torch.sin(th / 2)
    e = lambda a: torch.exp(1j * a.to(torch.complex64))
    m00 = e(-(phi + om) / 2) * c;  m01 = -e((phi - om) / 2) * sn
    m10 = e(-(phi - om) / 2) * sn; m11 = e((phi + om) / 2) * c
    return torch.stack([torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2)

def _sel_matrix(W, n):           # unitary of StronglyEntanglingLayers(imprimitive=CZ, default ranges)
    bits = (torch.arange(2 ** n, device=W.device)[:, None] >> (n - 1 - torch.arange(n, device=W.device))) & 1
    U = torch.eye(2 ** n, dtype=torch.complex64, device=W.device)
    R = _rot(W)
    for l in range(W.shape[0]):
        K = R[l, 0]
        for i in range(1, n):
            K = torch.kron(K, R[l, i])
        r = (l % (n - 1)) + 1
        cz = sum((bits[:, i] * bits[:, (i + r) % n]).float() for i in range(n if n > 2 else 1))
        U = (1 - 2 * (cz % 2)).to(torch.complex64)[:, None] * (K @ U)
    return U

def fast_forward(self, x):
    n, L = self.in_features, self.n_layer
    orig = list(x.shape[:-1]) + [-1]
    x = x.reshape(-1, n)
    w1, w2 = self.qnn.weights1, self.qnn.weights2
    z = (1 - 2 * ((torch.arange(2 ** n, device=x.device)[:, None] >> (n - 1 - torch.arange(n, device=x.device))) & 1)).to(x.dtype)
    phase = torch.exp(-0.5j * (x @ z.T).to(torch.complex64))
    psi = None
    for i in range(L):
        U = _sel_matrix(w1[i], n)
        psi = U[:, 0].expand(x.shape[0], -1) if psi is None else psi @ U.T
        psi = psi * phase
    psi = psi @ _sel_matrix(w2, n).T
    return ((psi.abs() ** 2) @ z).reshape(orig)

def make_fast_qiren(in_f=2, out_f=1):
    net = make_qiren(in_f, out_f)
    for m in net.modules():
        if isinstance(m, qiren_modules.QuantumLayer):
            m.forward = fast_forward.__get__(m)
    return net

# Self-check on this GPU against the official implementation (same weights, same inputs)
torch.manual_seed(0)
ref = make_qiren().to(dev).train(); fast = make_fast_qiren().to(dev).train()
fast.load_state_dict(ref.state_dict())
xt = (torch.rand(1, 500, 2, device=dev) * 2 - 1)
o1, _ = ref(xt); o1.pow(2).mean().backward()
o2, _ = fast(xt); o2.pow(2).mean().backward()
d_out = (o1 - o2).abs().max().item()
d_grad = max((a.grad - b.grad).abs().max().item() for a, b in zip(ref.parameters(), fast.parameters()))
print(f'Self-check: max output diff {d_out:.2e}, max gradient diff {d_grad:.2e}, parameters {sum(p.numel() for p in fast.parameters())}')
assert d_out < 1e-4 and d_grad < 1e-4, 'Fast implementation disagrees with the official one; stopping'
del ref, fast


import copy
# ---------- 4. Generic training loop: Adam, lr 1e-4 (SIREN settings); resumable; saves the best weights ----------
LR, SEED = 1e-4, 0
ROOT_SAVE = OUT / 'qiren_on_siren_tasks'

class QIRENWrap(torch.nn.Module):                   # wraps QIREN in SIREN's input/output format
    def __init__(self, in_f, out_f):
        super().__init__(); self.net = make_fast_qiren(in_f, out_f)
    def forward(self, inp):
        out, c = self.net(inp['coords'])
        return {'model_in': c, 'model_out': out}

def run_task(tag, net, batch_fn, steps, eval_fn=None, eval_every=None, report=None, print_every=500):
    """batch_fn() -> (coords, target); eval_fn(net) -> error, lower is better (if None, the training loss itself is used)"""
    SAVE = ROOT_SAVE / tag; SAVE.mkdir(parents=True, exist_ok=True)
    STATE, BEST = SAVE / 'resume_state.pth', SAVE / 'best_weights.pth'
    torch.manual_seed(SEED); np.random.seed(SEED)
    net = net.to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    step, best, best_step, hist = 0, float('inf'), -1, []
    best_sd = None
    if STATE.exists():
        st = torch.load(STATE, map_location=dev, weights_only=False)
        net.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        step, best, best_step, hist = st['step'], st['best'], st['best_step'], st['hist']
        torch.set_rng_state(st['rng'])
        if dev == 'cuda' and st.get('cuda_rng') is not None: torch.cuda.set_rng_state(st['cuda_rng'])
    rep = report or (lambda e: f'MSE {e:.6g}')
    if step >= steps:
        print(f'[{tag}] already finished: best {rep(best)} at step {best_step}'); return best, best_step

    def save_all():
        if best_sd is not None:
            torch.save(dict(model_state_dict=best_sd, step=best_step, error=best), BEST)
        tmp = STATE.with_suffix('.tmp')
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, best=best, best_step=best_step,
                        hist=hist, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if dev == 'cuda' else None), tmp)
        os.replace(tmp, STATE)

    print(f'===== [{tag}] starting at step {step} of {steps} =====', flush=True)
    t0, start = time.time(), step
    while step < steps:
        x, y = batch_fn()
        loss = ((net({'coords': x})['model_out'] - y) ** 2).mean()
        l = loss.item()
        if eval_fn is None and l < best:              # full data: this loss is the error of the current weights
            best, best_step = l, step
            best_sd = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            e = eval_fn(net)                           # weights after the update
            hist.append((step, e))
            if e < best:
                best, best_step = e, step
                best_sd = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        elif eval_fn is None and step % 100 == 0:
            hist.append((step, l))
        if step % print_every == 0 or step == steps or step - start == 5:
            if step % print_every == 0 or step == steps:
                save_all()
            per = (time.time() - t0) / (step - start)
            print(f'[{tag}] step {step}/{steps}  train MSE {l:.6g}  best {rep(best)} @ {best_step}'
                  f'  | {per:.3f}s/step, about {per * (steps - step) / 60:.0f} min left', flush=True)
    save_all()
    json.dump(dict(task=tag, steps=steps, lr=LR, seed=SEED, best_step=best_step, best_error=best,
                   best_report=rep(best)), open(SAVE / 'result.json', 'w'), indent=2)
    print(f'[{tag}] done: best {rep(best)} at step {best_step}', flush=True)
    return best, best_step

results = {}

# ---------- 5. Cat video: the same cat_video.mp4 used in our SIREN reproduction ----------
import glob, shutil, cv2
VDIR = OUT / 'data'; VDIR.mkdir(exist_ok=True)
vpath = VDIR / 'cat_video.mp4'                              # copy kept on Drive
if not vpath.exists():
    local = glob.glob('/content/siren_gdrive/cat_video.mp4') + glob.glob('/content/cat_video.mp4')
    if local:
        shutil.copy2(local[0], vpath)
    else:                                                   # otherwise download from the SIREN authors' shared folder
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
        files = gdown.download_folder('https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K',
                                      skip_download=True, quiet=True)
        f = next(f for f in files if os.path.basename(f.path).lower() == 'cat_video.mp4')
        gdown.download(id=f.id, output=str(vpath), quiet=False)
cap, frames = cv2.VideoCapture(str(vpath)), []              # same as dataio.Video for mp4: RGB, /255
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
cap.release()
vid = np.stack(frames).astype(np.float32) / 255.
T, H, W, C = vid.shape
print('Cat video:', vpath, vid.shape)
vid_t = torch.from_numpy(vid).to(dev)
data = (vid_t.reshape(-1, C) - 0.5) / 0.5                  # same as Implicit3DWrapper: [-1, 1]
NS = int(38e-4 * T * H * W)                               # same as SIREN sample_frac=38e-4

def to_coords(idx):                                        # same order and range as dataio.get_mgrid(dim=3)
    t, r, c = idx // (H * W), (idx // W) % H, idx % W
    return torch.stack([(t / max(T - 1, 1) - 0.5) * 2, (r / (H - 1) - 0.5) * 2, (c / (W - 1) - 0.5) * 2], -1).float()

def video_batch():                                         # random sampling each step, same as Implicit3DWrapper
    idx = torch.randint(0, T * H * W, (NS,), device=dev)
    return to_coords(idx)[None], data[idx][None]

FRAMES = [f for f in [0, 60, 120, 200] if f < T]           # the same 4 frames as SIREN's write_video_summary
fidx = torch.cat([torch.arange(f * H * W, (f + 1) * H * W, device=dev) for f in FRAMES])
fcoords, fgt = to_coords(fidx)[None], vid_t[FRAMES]

@torch.no_grad()
def video_eval(net):                                       # MSE on 4 frames ([0,1] pixels, prediction clipped to [0,1]), as in SIREN
    m = copy.deepcopy(net)                                 # copy, so evaluation does not change the BatchNorm statistics
    pred = (m({'coords': fcoords})['model_out'].reshape(len(FRAMES), H, W, C) / 2 + 0.5).clamp(0, 1)
    del m
    return torch.mean((fgt - pred) ** 2).item()

vpsnr = lambda e: f'PSNR {10 * np.log10(1 / e):.2f} dB'
results['video_cat__QIREN'] = run_task('video_cat__QIREN', QIRENWrap(3, 3), video_batch, steps=100000,
                                       eval_fn=video_eval, eval_every=1000, report=vpsnr, print_every=1000)

# ---------- 6. Summary: all finished QIREN-on-SIREN tasks ----------
print('\n===== Summary (best weights) =====')
for rj in sorted(ROOT_SAVE.glob('*/result.json')):
    r = json.load(open(rj))
    print(f"{rj.parent.name:32s}  {r.get('best_report', r.get('best_psnr'))}  at step {r['best_step']}")
try:                                                       # our earlier run of SIREN's official script on the cat video
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for name in ['video_cat_sine', 'video_cat_relu']:
        ev = glob.glob(f'/content/drive/MyDrive/siren_reproduction/logs/{name}/summaries/*')
        ea = EventAccumulator(ev[0]); ea.Reload(); ps = [e.value for e in ea.Scalars('train_psnr')]
        print(f'{name} reference (our earlier reproduction): last PSNR {ps[-1]:.2f} dB, best {max(ps):.2f} dB, logged up to step {ea.Scalars("train_psnr")[-1].step}')
except Exception as ex:
    print('Could not read the earlier SIREN video logs:', ex)
