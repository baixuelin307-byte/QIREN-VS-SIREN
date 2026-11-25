# ===== 检查三组实验的所有结果，并整理成可以直接上传 GitHub 的文件夹（只读取和复制，不训练）=====
import os, json, shutil, glob
from pathlib import Path
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

Q = Path('/content/drive/MyDrive/QIREN_paper_recheck')      # QIREN 实验（实验 1、2、3）
S = Path('/content/drive/MyDrive/siren_reproduction')       # SIREN 原论文复现
REPO = Path('/content/drive/MyDrive/qiren_siren_repo')      # 整理好的仓库文件夹
MAX_MB = 20                                                 # 超过 20 MB 的文件不放进仓库（GitHub 单文件上限 100 MB）

# ---------- 1. 必须存在的关键结果 ----------
REQUIRED = {
    '实验 1：QIREN 复现 vs 论文': [
        Q / 'baseline_best_epoch' / 'summary_best_epoch_all_models.csv',
        Q / 'best_epoch_runs' / 'qiren_best_vs_paper.csv',
        Q / 'best_epoch_runs' / 'best_vs_last_all_seeds.csv',
        Q / 'environment.json',
    ],
    '实验 2：QIREN vs SIREN（小网络）': [
        Q / 'siren_selected_hyperparams.json',
        Q / 'siren_tune',
        Q / 'siren_repo_default',
    ],
    '实验 3：QIREN 跑 SIREN 原任务': [
        Q / 'qiren_on_siren_tasks' / 'image_fitting_gpu' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'audio_gt_bach__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'audio_gt_counting__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'video_cat__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'test_results.csv',
        Q / 'qiren_on_siren_tasks' / 'test_figure.png',
    ],
    'SIREN 原论文复现': [S],
}
print('================ 检查关键结果 ================')
missing = 0
for exp, items in REQUIRED.items():
    print(f'\n【{exp}】')
    for p in items:
        ok = p.exists()
        missing += not ok
        extra = f'（{len(list(p.iterdir()))} 个文件）' if ok and p.is_dir() else ''
        print(f"  {'✅' if ok else '❌ 缺少'}  {p}{extra}")
print(f'\n缺少 {missing} 项' if missing else '\n全部关键结果都在 ✅')

# ---------- 2. 复制到仓库结构 ----------
def copy_matching(src_dir, dst_dir, patterns, exclude=()):
    n = 0
    for pat in patterns:
        for f in glob.glob(str(src_dir / pat), recursive=True):
            f = Path(f)
            if not f.is_file() or any(x in f.parts for x in exclude): continue
            if f.stat().st_size > MAX_MB * 1e6: print(f'  跳过（太大）: {f}'); continue
            dst = dst_dir / f.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst); n += 1
    return n

if REPO.exists(): shutil.rmtree(REPO)
plan = [
    ('results/exp1_qiren_vs_paper',      Q,                          ['baseline_best_epoch/*.csv', 'best_epoch_runs/*.csv',
                                                                      'runs/*.json', 'environment.json'], ()),
    ('results/exp2_qiren_vs_siren',      Q,                          ['siren_selected_hyperparams.json', 'siren_tune/**/*.csv',
                                                                      'siren_tune/**/*.json', 'siren_repo_default/**/*.csv',
                                                                      'siren_repo_default/**/*.json',
                                                                      'siren_grid_vs_paper/*.csv'], ()),
    ('results/exp3_qiren_on_siren_tasks', Q / 'qiren_on_siren_tasks', ['*/result.json', 'test_results.csv',
                                                                      '*/best_weights.pth'], ('video_cat__SIREN',)),
    ('figures',                          Q,                          ['*.png', 'qiren_on_siren_tasks/*.png'], ()),
    ('results/siren_reproduction',       S,                          ['**/*.csv', '**/*.json', '**/*.png'], ('logs',)),
]
print('\n================ 整理到仓库文件夹 ================')
for dst, src, pats, excl in plan:
    n = copy_matching(src, REPO / dst, pats, excl) if src.exists() else 0
    print(f"  {dst:40s} {n} 个文件{'' if n else '  ← 没有找到文件'}")

# 实验 3 的结果表直接打印出来（发给 Claude 填 README）
t = Q / 'qiren_on_siren_tasks' / 'test_results.csv'
if t.exists():
    print('\n================ 实验 3 测试结果 ================')
    print(t.read_text())

# ---------- 3. 仓库目录树 + 打包 ----------
print('================ 仓库目录 ================')
for d in sorted({p.parent for p in REPO.rglob('*') if p.is_file()}):
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    shown = ', '.join(files[:6]) + (f' …（共 {len(files)} 个）' if len(files) > 6 else '')
    print(f'{d.relative_to(REPO)}/  {shown}')
z = shutil.make_archive(str(REPO), 'zip', REPO)
print(f'\n已打包: {z}（{os.path.getsize(z) / 1e6:.1f} MB），下载后解压，把 notebooks 和 README.md 放进去即可上传')
