# ===== Check all results of the three experiments and package them into a folder ready for GitHub (read and copy only, no training) =====
import os, json, shutil, glob
from pathlib import Path
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

Q = Path('/content/drive/MyDrive/QIREN_paper_recheck')      # QIREN experiments (Experiments 1, 2, 3)
S = Path('/content/drive/MyDrive/siren_reproduction')       # SIREN paper reproduction
REPO = Path('/content/drive/MyDrive/qiren_siren_repo')      # packaged repository folder
MAX_MB = 20                                                 # files over 20 MB are left out (GitHub limit is 100 MB per file)

# ---------- 1. Key results that must exist ----------
REQUIRED = {
    'Experiment 1: QIREN reproduction vs. paper': [
        Q / 'baseline_best_epoch' / 'summary_best_epoch_all_models.csv',
        Q / 'best_epoch_runs' / 'qiren_best_vs_paper.csv',
        Q / 'best_epoch_runs' / 'best_vs_last_all_seeds.csv',
        Q / 'environment.json',
    ],
    'Experiment 2: QIREN vs. SIREN (small networks)': [
        Q / 'siren_selected_hyperparams.json',
        Q / 'siren_tune',
        Q / 'siren_repo_default',
    ],
    'Experiment 3: QIREN on SIREN tasks': [
        Q / 'qiren_on_siren_tasks' / 'image_fitting_gpu' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'audio_gt_bach__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'audio_gt_counting__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'video_cat__QIREN' / 'result.json',
        Q / 'qiren_on_siren_tasks' / 'test_results.csv',
        Q / 'qiren_on_siren_tasks' / 'test_figure.png',
    ],
    'SIREN paper reproduction': [S],
}
print('================ Checking key results ================')
missing = 0
for exp, items in REQUIRED.items():
    print(f'\n[{exp}]')
    for p in items:
        ok = p.exists()
        missing += not ok
        extra = f' ({len(list(p.iterdir()))} files)' if ok and p.is_dir() else ''
        print(f"  {'✅' if ok else '❌ missing'}  {p}{extra}")
print(f'\n{missing} item(s) missing' if missing else '\nAll key results present ✅')

# ---------- 2. Copy into the repository layout ----------
def copy_matching(src_dir, dst_dir, patterns, exclude=()):
    n = 0
    for pat in patterns:
        for f in glob.glob(str(src_dir / pat), recursive=True):
            f = Path(f)
            if not f.is_file() or any(x in f.parts for x in exclude): continue
            if f.stat().st_size > MAX_MB * 1e6: print(f'  skipped (too large): {f}'); continue
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
print('\n================ Copying into the repository folder ================')
for dst, src, pats, excl in plan:
    n = copy_matching(src, REPO / dst, pats, excl) if src.exists() else 0
    print(f"  {dst:40s} {n} files{'' if n else '  <- no files found'}")

# Print the Experiment 3 results table
t = Q / 'qiren_on_siren_tasks' / 'test_results.csv'
if t.exists():
    print('\n================ Experiment 3 test results ================')
    print(t.read_text())

# ---------- 3. Directory tree + zip ----------
print('================ Repository layout ================')
for d in sorted({p.parent for p in REPO.rglob('*') if p.is_file()}):
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    shown = ', '.join(files[:6]) + (f' ... ({len(files)} in total)' if len(files) > 6 else '')
    print(f'{d.relative_to(REPO)}/  {shown}')
z = shutil.make_archive(str(REPO), 'zip', REPO)
print(f'\nPackaged: {z} ({os.path.getsize(z) / 1e6:.1f} MB). Unzip it and add the notebooks and README.md before uploading.')
