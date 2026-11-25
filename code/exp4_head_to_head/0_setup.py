# ===== Experiment 4 setup: restore QIREN code, data loaders and model builder (run this before the Experiment 4 cell) =====
import os, sys, subprocess
from pathlib import Path
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

OUT = Path('/content/drive/MyDrive/QIREN_paper_recheck')
REPO_SHA = '56ab4010ee3fbb3371c9d5d5f5284a8c41d4e6f3'
ROOT = Path('/content/QIREN_exp4_repo')                     # separate clean copy, independent of other cells
QINR = ROOT / 'qinr'
assert (OUT / 'best_epoch_runs').is_dir() and (OUT / 'baseline_best_epoch').is_dir(), f'Best-weight folders not found in {OUT}'

if not (ROOT / '.git').is_dir():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/GGorMM1/QIREN.git', str(ROOT)], check=True)
subprocess.run(['git', '-C', str(ROOT), 'checkout', '-q', '-f', REPO_SHA], check=True)
subprocess.run(['git', '-C', str(ROOT), 'checkout', '-q', '--', '.'], check=True)    # start from unpatched files
try:
    import pennylane
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pennylane'], check=True)

def patch(relpath, replacements):                          # same patches as in the reproduction notebook
    p = QINR / relpath
    text = p.read_text(encoding='utf-8').replace('\r\n', '\n')
    for before, after in replacements:
        assert text.count(before) == 1, f'{relpath}: anchor not found: {before!r}'
        text = text.replace(before, after, 1)
    p.write_text(text, encoding='utf-8')

patch('modules.py', [
    ('import qiskit.providers.aer.noise as noise',
     'try:\n    import qiskit.providers.aer.noise as noise\nexcept ImportError:\n    noise = None'),
    ('qml.RZ(inputs[j], wires=j)', 'qml.RZ(inputs[..., j], wires=j)'),
])
patch('image/data_process.py', [
    ('def get_cameraman_tensor(sidelength):', "def get_cameraman_tensor(sidelength, name='coffee'):"),
    ('    img = Image.fromarray(skimage.color.rgb2gray(skimage.data.coffee()))\n',
     "    if name == 'camera':\n        img = Image.fromarray(skimage.data.camera())\n"
     "    elif name == 'astronaut':\n        img = Image.fromarray(skimage.color.rgb2gray(skimage.data.astronaut()))\n"
     "    elif name == 'coffee':\n        img = Image.fromarray(skimage.color.rgb2gray(skimage.data.coffee()))\n"
     "    else:\n        raise ValueError(f'Unknown image: {name}')\n"),
    ('    def __init__(self, sidelength):', "    def __init__(self, sidelength, name='coffee'):"),
    ('        img = get_cameraman_tensor(sidelength)', '        img = get_cameraman_tensor(sidelength, name)'),
])

os.chdir(QINR)
sys.path = [str(QINR)] + [p for p in sys.path if 'siren' not in p.lower()]
for m in [m for m in sys.modules if m in ('modules', 'image', 'image.data_process')]:
    del sys.modules[m]                                       # drop any other "modules" imported earlier (e.g. SIREN's)
from modules import Hybridren, Siren
from image.data_process import ImageFitting

SEEDS = (0, 1, 2, 3, 4)

def make_model(model, in_f, omega=1):                       # identical to the reproduction notebook
    if model == 'hybridren':
        return Hybridren(in_features=in_f, out_features=1, hidden_features=8, hidden_layers=2, spectrum_layer=2, use_noise=0)
    if model == 'siren':
        return Siren(in_features=in_f, out_features=1, hidden_features=10, hidden_layers=6,
                     first_omega_0=omega, hidden_omega_0=omega)
    raise ValueError(model)

assert sum(p.numel() for p in make_model('hybridren', 2).parameters()) == 657
assert sum(p.numel() for p in make_model('siren', 2).parameters()) == 701
print('Ready for Experiment 4 ✅  (QIREN 657 params, SIREN 701 params)')
