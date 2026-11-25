# ===== QIREN 跑 SIREN 原论文的音频任务（Bach、Counting），并用同一训练循环训练 SIREN 官方音频网络作对照 =====
# GPU 快速精确版；每个任务可断点续跑；保存最好权重
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

# ---------- 4. SIREN 官方仓库（数据、SIREN 模型都用它的）----------
SIREN = '/content/siren'
if not os.path.exists(SIREN):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/vsitzmann/siren.git', SIREN], check=True)
    subprocess.run(f"sed -i 's/Image.ANTIALIAS/Image.LANCZOS/' {SIREN}/dataio.py", shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cmapy', 'scikit-video', 'opencv-python',
                'h5py', 'ordered-set', 'gdown'], check=True)
if SIREN not in sys.path: sys.path.append(SIREN)
import dataio
import modules as siren_modules                     # SIREN 的 modules.py（QIREN 的已用别名导入，不冲突）

# ---------- 5. 通用训练循环：Adam、lr 1e-4（SIREN 设置）；断点续跑；保存最好权重 ----------
LR, SEED = 1e-4, 0
ROOT_SAVE = OUT / 'qiren_on_siren_tasks'

class QIRENWrap(torch.nn.Module):                   # 包成 SIREN 的输入输出格式
    def __init__(self, in_f, out_f):
        super().__init__(); self.net = make_fast_qiren(in_f, out_f)
    def forward(self, inp):
        out, c = self.net(inp['coords'])
        return {'model_in': c, 'model_out': out}

def run_task(tag, net, batch_fn, steps, eval_fn=None, eval_every=None, report=None, print_every=500):
    """batch_fn() -> (coords, target)；eval_fn(net) -> 越小越好的误差（为 None 时用训练 loss 本身）"""
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
        print(f'[{tag}] 已完成：最好 {rep(best)} @ 第 {best_step} 步'); return best, best_step

    def save_all():
        if best_sd is not None:
            torch.save(dict(model_state_dict=best_sd, step=best_step, error=best), BEST)
        tmp = STATE.with_suffix('.tmp')
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, best=best, best_step=best_step,
                        hist=hist, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if dev == 'cuda' else None), tmp)
        os.replace(tmp, STATE)

    print(f'===== [{tag}] 从第 {step} 步开始，共 {steps} 步 =====', flush=True)
    t0, start = time.time(), step
    while step < steps:
        x, y = batch_fn()
        loss = ((net({'coords': x})['model_out'] - y) ** 2).mean()
        l = loss.item()
        if eval_fn is None and l < best:              # 全量数据：这个 loss 就是当前权重的误差
            best, best_step = l, step
            best_sd = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            e = eval_fn(net)                           # 更新后的权重
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
                  f'  | {per:.3f}s/步，剩余约 {per * (steps - step) / 60:.0f} 分钟', flush=True)
    save_all()
    json.dump(dict(task=tag, steps=steps, lr=LR, seed=SEED, best_step=best_step, best_error=best,
                   best_report=rep(best)), open(SAVE / 'result.json', 'w'), indent=2)
    print(f'[{tag}] 完成：最好 {rep(best)} @ 第 {best_step} 步', flush=True)
    return best, best_step

results = {}

# ---------- 6. 音频（SIREN train_audio.py：整段信号一个 batch，9001 步，lr 1e-4，坐标 [-100,100]）----------
# QIREN 和 SIREN 官方音频网络（sine, mlp）用同一个训练循环跑，方便直接对比
for wav in ['gt_bach', 'gt_counting']:
    inp, gtd = dataio.ImplicitAudioWrapper(dataio.AudioFile(filename=f'{SIREN}/data/{wav}.wav'))[0]
    xa, ya = inp['coords'][None].to(dev), gtd['func'][None].to(dev)
    batch = lambda xa=xa, ya=ya: (xa, ya)
    for mname, builder in [('QIREN', lambda: QIRENWrap(1, 1)),
                           ('SIREN', lambda: siren_modules.SingleBVPNet(type='sine', mode='mlp', in_features=1))]:
        results[f'{wav}__{mname}'] = run_task(f'audio_{wav}__{mname}', builder(), batch, steps=9001)

# ---------- 8. 汇总 ----------
print('\n===== 汇总（最好权重）=====')
for k, (err, st) in results.items():
    print(f'{k:28s}  ' + (vpsnr(err) if k.startswith('video') else f'MSE {err:.6g}') + f'  @ 第 {st} 步')
try:                                                       # (video is trained in 3_train_video.py)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = glob.glob('/content/drive/MyDrive/siren_reproduction/logs/video_cat_sine/summaries/*')
    ea = EventAccumulator(ev[0]); ea.Reload(); ps = [e.value for e in ea.Scalars('train_psnr')]
    print(f'对照 SIREN 视频（你之前的复现）：最后 PSNR {ps[-1]:.2f} dB，最高 {max(ps):.2f} dB')
except Exception as ex:
    print('没读到之前 SIREN 视频的日志：', ex)
