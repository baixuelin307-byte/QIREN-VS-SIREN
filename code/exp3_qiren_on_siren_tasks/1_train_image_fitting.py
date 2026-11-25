# ===== QIREN on SIREN's image fitting task (fast exact GPU simulation; resumable; saves the best weights) =====
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

def make_qiren():
    return qiren_modules.Hybridren(in_features=2, out_features=1, hidden_features=8,
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

def make_fast_qiren():
    net = make_qiren()
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

# ---------- 4. Image data from the official SIREN repository (cameraman, 256×256, same as our SIREN runs) ----------
SIREN = '/content/siren'
if not os.path.exists(SIREN):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/vsitzmann/siren.git', SIREN], check=True)
    subprocess.run(f"sed -i 's/Image.ANTIALIAS/Image.LANCZOS/' {SIREN}/dataio.py", shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cmapy', 'scikit-video', 'opencv-python'], check=True)
if SIREN not in sys.path: sys.path.append(SIREN)
import dataio
model_input, gt = next(iter(DataLoader(dataio.Implicit2DWrapper(dataio.Camera(), sidelength=256), batch_size=1)))
x, y = model_input['coords'].to(dev), gt['img'].to(dev)

# ---------- 5. Training with SIREN's settings: Adam, lr 1e-4, 15,000 steps, full image as one batch ----------
STEPS, LR, SEED, SAVE_EVERY = 15000, 1e-4, 0, 200
SAVE = OUT / 'qiren_on_siren_tasks' / 'image_fitting_gpu'; SAVE.mkdir(parents=True, exist_ok=True)
STATE, BEST = SAVE / 'resume_state.pth', SAVE / 'best_weights.pth'
def psnr(mse): return 10 * np.log10(1 / (mse / 4))   # as in SIREN: pixels mapped to [0,1]

torch.manual_seed(SEED); np.random.seed(SEED)
net = make_fast_qiren().to(dev).train()
opt = torch.optim.Adam(net.parameters(), lr=LR)
step, best, best_step, losses, best_sd = 0, float('inf'), -1, [], None
if STATE.exists():
    st = torch.load(STATE, map_location=dev, weights_only=False)
    net.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    step, best, best_step, losses = st['step'], st['best'], st['best_step'], st['losses']
    print(f'Resuming from step {step}, best PSNR so far {psnr(best):.2f} dB (step {best_step})')

def save_all():
    if best_sd is not None:
        torch.save(dict(model_state_dict=best_sd, step=best_step, mse=best, psnr=psnr(best)), BEST)
    tmp = STATE.with_suffix('.tmp')
    torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, best=best,
                    best_step=best_step, losses=losses), tmp)
    os.replace(tmp, STATE)

t0, start = time.time(), step
while step < STEPS:
    out, _ = net(x)
    loss = ((out - y) ** 2).mean()
    l = loss.item(); losses.append(l)
    if l < best:                                       # this loss belongs to the current weights: record before the update
        best, best_step = l, step
        best_sd = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
    opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % SAVE_EVERY == 0 or step == STEPS or step - start == 5:
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_all()
        per = (time.time() - t0) / (step - start)
        print(f'step {step}/{STEPS}  PSNR {psnr(l):.2f} dB  best {psnr(best):.2f} dB @ {best_step}'
              f'  | {per:.3f}s/step, about {per * (STEPS - step) / 60:.0f} min left', flush=True)

save_all()
res = dict(task='image_fitting (SIREN, cameraman 256x256)', steps=STEPS, lr=LR, seed=SEED,
           best_step=best_step, best_mse=best, best_psnr=psnr(best),
           psnr_last1000_mean=float(np.mean([psnr(v) for v in losses[-1000:]])))
json.dump(res, open(SAVE / 'result.json', 'w'), indent=2)
print('\nDone:', res)
print('SIREN reference (our reproduction, same task): mean PSNR over the last 1,000 steps about 56.7 dB; ReLU 25.8, Tanh 26.1, ReLU P.E. 43.3')
