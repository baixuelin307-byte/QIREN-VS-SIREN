# ===== QIREN 跑 SIREN 原论文的图像拟合任务（GPU 快速精确版；可断点续跑；保存最好权重）=====
import os, sys, time, json, copy, subprocess, importlib.util
from pathlib import Path

# ---------- 1. Drive ----------
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/QIREN_paper_recheck'); OUT.mkdir(parents=True, exist_ok=True)

# ---------- 2. QIREN 官方代码（固定 commit）+ 必要补丁 ----------
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
print('设备:', torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU')

def make_qiren():
    return qiren_modules.Hybridren(in_features=2, out_features=1, hidden_features=8,
                                   hidden_layers=2, spectrum_layer=2, use_noise=0)

# ---------- 3. 量子层的精确快速实现（与官方 PennyLane 电路数学上相同）----------
#   官方电路：|psi> = SEL(w2) · D(x) · SEL(w1[1]) · D(x) · SEL(w1[0]) |0>，输出 <Z_i>
#   SEL 只依赖权重，D(x) = RZ(x_0)⊗...⊗RZ(x_7) 是对角相位，所以可以写成矩阵乘法
def _rot(w):                     # qml.Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
    phi, th, om = w[..., 0], w[..., 1], w[..., 2]
    c, sn = torch.cos(th / 2), torch.sin(th / 2)
    e = lambda a: torch.exp(1j * a.to(torch.complex64))
    m00 = e(-(phi + om) / 2) * c;  m01 = -e((phi - om) / 2) * sn
    m10 = e(-(phi - om) / 2) * sn; m11 = e((phi + om) / 2) * c
    return torch.stack([torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2)

def _sel_matrix(W, n):           # StronglyEntanglingLayers(imprimitive=CZ, 默认 ranges) 的酉矩阵
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

# 自检：在你的 GPU 上和官方实现逐项对比（同一份权重、同一批输入）
torch.manual_seed(0)
ref = make_qiren().to(dev).train(); fast = make_fast_qiren().to(dev).train()
fast.load_state_dict(ref.state_dict())
xt = (torch.rand(1, 500, 2, device=dev) * 2 - 1)
o1, _ = ref(xt); o1.pow(2).mean().backward()
o2, _ = fast(xt); o2.pow(2).mean().backward()
d_out = (o1 - o2).abs().max().item()
d_grad = max((a.grad - b.grad).abs().max().item() for a, b in zip(ref.parameters(), fast.parameters()))
print(f'自检：输出最大差 {d_out:.2e}，梯度最大差 {d_grad:.2e}，参数量 {sum(p.numel() for p in fast.parameters())}')
assert d_out < 1e-4 and d_grad < 1e-4, '快速实现与官方实现不一致，停止'
del ref, fast

# ---------- 4. SIREN 官方仓库的图像数据（cameraman，256×256，与你跑 SIREN 时相同）----------
SIREN = '/content/siren'
if not os.path.exists(SIREN):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/vsitzmann/siren.git', SIREN], check=True)
    subprocess.run(f"sed -i 's/Image.ANTIALIAS/Image.LANCZOS/' {SIREN}/dataio.py", shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cmapy', 'scikit-video', 'opencv-python'], check=True)
if SIREN not in sys.path: sys.path.append(SIREN)
import dataio
model_input, gt = next(iter(DataLoader(dataio.Implicit2DWrapper(dataio.Camera(), sidelength=256), batch_size=1)))
x, y = model_input['coords'].to(dev), gt['img'].to(dev)

# ---------- 5. 训练：SIREN 设置 = Adam, lr 1e-4, 15,000 步, 全图一个 batch ----------
STEPS, LR, SEED, SAVE_EVERY = 15000, 1e-4, 0, 200
SAVE = OUT / 'qiren_on_siren_tasks' / 'image_fitting_gpu'; SAVE.mkdir(parents=True, exist_ok=True)
STATE, BEST = SAVE / 'resume_state.pth', SAVE / 'best_weights.pth'
def psnr(mse): return 10 * np.log10(1 / (mse / 4))   # 与 SIREN 一致：像素映射到 [0,1]

torch.manual_seed(SEED); np.random.seed(SEED)
net = make_fast_qiren().to(dev).train()
opt = torch.optim.Adam(net.parameters(), lr=LR)
step, best, best_step, losses, best_sd = 0, float('inf'), -1, [], None
if STATE.exists():
    st = torch.load(STATE, map_location=dev, weights_only=False)
    net.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    step, best, best_step, losses = st['step'], st['best'], st['best_step'], st['losses']
    print(f'从第 {step} 步继续，目前最好 PSNR {psnr(best):.2f} dB（第 {best_step} 步）')

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
    if l < best:                                       # 这个 loss 对应当前权重，先记下再更新
        best, best_step = l, step
        best_sd = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
    opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % SAVE_EVERY == 0 or step == STEPS or step - start == 5:
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_all()
        per = (time.time() - t0) / (step - start)
        print(f'step {step}/{STEPS}  PSNR {psnr(l):.2f} dB  best {psnr(best):.2f} dB @ {best_step}'
              f'  | {per:.3f}s/步，剩余约 {per * (STEPS - step) / 60:.0f} 分钟', flush=True)

save_all()
res = dict(task='image_fitting (SIREN, cameraman 256x256)', steps=STEPS, lr=LR, seed=SEED,
           best_step=best_step, best_mse=best, best_psnr=psnr(best),
           psnr_last1000_mean=float(np.mean([psnr(v) for v in losses[-1000:]])))
json.dump(res, open(SAVE / 'result.json', 'w'), indent=2)
print('\n完成：', res)
print('对照 SIREN（你的复现，同一任务）：最后 1,000 步平均 PSNR 约 56.7 dB；ReLU 25.8，Tanh 26.1，ReLU P.E. 43.3')
