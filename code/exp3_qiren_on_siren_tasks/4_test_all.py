# ===== 测试：用最好权重，按 SIREN 原论文的测试方式评估 QIREN（和 SIREN）=====
# 只测试，不训练
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

def make_qiren(in_f=2, out_f=1):
    return qiren_modules.Hybridren(in_features=in_f, out_features=out_f, hidden_features=8,
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

def make_fast_qiren(in_f=2, out_f=1):
    net = make_qiren(in_f, out_f)
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


import re
# ---------- 4. SIREN 官方仓库（测试数据、SIREN 模型都用它的）----------
SIREN = '/content/siren'
if not os.path.exists(SIREN):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/vsitzmann/siren.git', SIREN], check=True)
    subprocess.run(f"sed -i 's/Image.ANTIALIAS/Image.LANCZOS/' {SIREN}/dataio.py", shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cmapy', 'scikit-video', 'opencv-python',
                'h5py', 'ordered-set'], check=True)
if SIREN not in sys.path: sys.path.append(SIREN)
import dataio
import modules as siren_modules
import cv2, glob, math
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = OUT / 'qiren_on_siren_tasks'
SLOG = Path('/content/drive/MyDrive/siren_reproduction/logs')
rows, figs = [], {}

def load_qiren(path, in_f, out_f):
    sd = torch.load(path, map_location=dev, weights_only=False)['model_state_dict']
    if any(k.startswith('net.net.') for k in sd):            # 训练时外面包了一层 wrapper
        sd = {k[4:]: v for k, v in sd.items()}
    net = make_fast_qiren(in_f, out_f).to(dev).train()      # 与训练相同：BatchNorm 用批统计
    net.load_state_dict(sd); return net

def psnr(mse): return 10 * math.log10(1 / mse)             # 像素在 [0,1]

# ---------- 5. 图像拟合：整张 256×256 cameraman，PSNR（预测截断到 [0,1]，与 SIREN 相同）----------
p = R / 'image_fitting_gpu' / 'best_weights.pth'
if p.exists():
    inp, gt = dataio.Implicit2DWrapper(dataio.Camera(), sidelength=256)[0]
    x, y = inp['coords'][None].to(dev), gt['img'][None].to(dev)
    with torch.no_grad():
        out, _ = load_qiren(p, 2, 1)(x)                     # 整张图一个 batch，与训练相同
    pred, g = (out / 2 + 0.5).clamp(0, 1), y / 2 + 0.5
    rows.append(dict(task='Image fitting (cameraman 256×256)', model='QIREN', metric='PSNR (dB)',
                     value=round(psnr(((pred - g) ** 2).mean().item()), 2), note='full image'))
    figs['image'] = (g[0].reshape(256, 256).cpu().numpy(), pred[0].reshape(256, 256).cpu().numpy())
else:
    print('没找到图像拟合的最好权重:', p)

# ---------- 6. 音频：整段信号 MSE（与 SIREN train_audio.py 的损失相同）----------
for wav in ['gt_bach', 'gt_counting']:
    inp, gt = dataio.ImplicitAudioWrapper(dataio.AudioFile(filename=f'{SIREN}/data/{wav}.wav'))[0]
    x, y = inp['coords'][None].to(dev), gt['func'][None].to(dev)
    for mname in ['QIREN', 'SIREN']:
        p = R / f'audio_{wav}__{mname}' / 'best_weights.pth'
        if not p.exists():
            print('没找到:', p); continue
        with torch.no_grad():
            if mname == 'QIREN':
                out, _ = load_qiren(p, 1, 1)(x)
            else:
                net = siren_modules.SingleBVPNet(type='sine', mode='mlp', in_features=1).to(dev).eval()
                net.load_state_dict(torch.load(p, map_location=dev, weights_only=False)['model_state_dict'])
                out = net({'coords': x})['model_out']
        rows.append(dict(task=f'Audio: {wav[3:].capitalize()}', model=mname, metric='MSE',
                         value=float(f'{((out - y) ** 2).mean().item():.4e}'), note='full signal'))

# ---------- 7. 视频：所有帧逐帧 PSNR，报告均值 ± 标准差（SIREN 论文的报告方式）----------
vpath = OUT / 'data' / 'cat_video.mp4'
cap, frames = cv2.VideoCapture(str(vpath)), []
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
cap.release()
vid = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.).to(dev)
T, H, W, C = vid.shape; N = T * H * W
print('cat 视频:', tuple(vid.shape))

def coords_of(idx):                                        # 与 dataio.get_mgrid(dim=3) 相同
    t, r, c = idx // (H * W), (idx // W) % H, idx % W
    return torch.stack([(t / max(T - 1, 1) - 0.5) * 2, (r / (H - 1) - 0.5) * 2, (c / (W - 1) - 0.5) * 2], -1).float()

def frame_psnr(pred):                                      # pred: (N, C) in [-1, 1]
    p01 = (pred.reshape(T, H, W, C) / 2 + 0.5).clamp(0, 1)
    mse = ((p01 - vid) ** 2).mean((1, 2, 3))
    return (10 * torch.log10(1 / mse)).cpu().numpy(), p01

video_models = []
p = R / 'video_cat__QIREN' / 'best_weights.pth'
if p.exists(): video_models.append(('QIREN', p))
for typ in ['sine', 'relu']:
    ck = SLOG / f'video_cat_{typ}' / 'checkpoints'
    cands = [ck / 'model_final.pth'] if (ck / 'model_final.pth').exists() else \
            sorted(ck.glob('model_epoch_*.pth'), key=lambda q: int(re.findall(r'\d+', q.stem)[-1]))[-1:]
    if cands: video_models.append((f'SIREN official ({typ})', cands[0]))
    else: print('没找到 SIREN 视频权重:', ck)

for mname, p in video_models:
    pred = torch.empty(N, C, device=dev)
    with torch.no_grad():
        if mname == 'QIREN':
            # QIREN 的 BatchNorm 用批统计：按训练时的方式，把全部像素随机打乱后分成和训练 batch 一样大的块
            net = load_qiren(p, 3, 3)
            NS = int(38e-4 * N)
            perm = torch.randperm(N, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
            for idx in torch.chunk(perm, math.ceil(N / NS)):
                out, _ = net(coords_of(idx)[None]); pred[idx] = out[0]
        else:
            typ = 'sine' if 'sine' in mname else 'relu'
            net = siren_modules.SingleBVPNet(type=typ, in_features=3, out_features=C, mode='mlp',
                                             hidden_features=1024, num_hidden_layers=3).to(dev).eval()
            net.load_state_dict(torch.load(p, map_location=dev, weights_only=False))
            for s0 in range(0, N, 2 ** 19):
                idx = torch.arange(s0, min(s0 + 2 ** 19, N), device=dev)
                pred[idx] = net({'coords': coords_of(idx)[None]})['model_out'][0]
    ps, p01 = frame_psnr(pred)
    rows.append(dict(task='Video: Cat', model=mname, metric='PSNR (dB), all frames',
                     value=f'{ps.mean():.2f} ± {ps.std():.2f}', note=f'{p.name}; {T} frames'))
    figs[mname] = p01[[0, T // 2, T - 1]].cpu().numpy()
    del pred, p01; torch.cuda.empty_cache() if dev == 'cuda' else None

# ---------- 8. 结果表 + 图 ----------
df = pd.DataFrame(rows)
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 60)
print('\n===== 测试结果（最好权重，按 SIREN 的测试方式）=====')
print(df.to_string(index=False))
print('\n对照 SIREN 图像拟合（你的复现）：SIREN 56.74 dB，ReLU P.E. 43.25，RBF-ReLU 31.26，Tanh 26.05，ReLU 25.83')
df.to_csv(R / 'test_results.csv', index=False)

vmods = [m for m in figs if m != 'image']
ncol = 1 + len(vmods)
fig, ax = plt.subplots(3 + ('image' in figs), ncol, figsize=(3.2 * ncol, 3.2 * (3 + ('image' in figs))), squeeze=False)
ri = 0
if 'image' in figs:
    ax[0, 0].imshow(figs['image'][0], cmap='gray', vmin=0, vmax=1); ax[0, 0].set_title('Cameraman GT')
    ax[0, 1].imshow(figs['image'][1], cmap='gray', vmin=0, vmax=1)
    ax[0, 1].set_title(f"QIREN {df[df.task.str.startswith('Image')].value.iloc[0]} dB")
    ri = 1
gtf = vid[[0, T // 2, T - 1]].cpu().numpy()
for r, fi in enumerate([0, T // 2, T - 1]):
    ax[ri + r, 0].imshow(gtf[r]); ax[ri + r, 0].set_title(f'Cat GT, frame {fi}')
    for j, m in enumerate(vmods, 1):
        ax[ri + r, j].imshow(figs[m][r]); ax[ri + r, j].set_title(f'{m}, frame {fi}', fontsize=9)
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); fp = R / 'test_figure.png'; plt.savefig(fp, dpi=120)
print('\n结果表:', R / 'test_results.csv'); print('图:', fp)
try:
    from IPython.display import Image as _I, display; display(_I(str(fp)))
except Exception:
    pass
