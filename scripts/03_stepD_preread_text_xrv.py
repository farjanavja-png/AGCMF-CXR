# =============================================================================
# AGCMF_Rev6_StepD_PreReadText_XRV.py
#
# CLINICALLY REALISTIC, NON-CIRCULAR TASK (answers Reviewer: labels are
# extracted from the same report text). Two changes vs Step 1, applied
# IDENTICALLY to all seven models M1–M7:
#
#  1. TEXT = information available BEFORE the radiologist reads this image:
#       - INDICATION / HISTORY / REASON FOR EXAM section of the current report
#       - IMPRESSION (else FINDINGS) of the patient's most recent PRIOR report
#     The current report's FINDINGS / IMPRESSION / WET READ are NEVER used.
#  2. IMAGE ENCODER = TorchXRayVision DenseNet-121 pretrained on NIH
#     ChestX-ray14 ONLY (no MIMIC-CXR, no CheXpert in pretraining ->
#     no overlap with our test set or our external set).
#
# Everything else (split, burden score, labels, epochs, seed, optimiser,
# evaluation, statistics, calibration, MC-Dropout, CheXpert) = Step 1.
# Results go to a NEW folder: /content/drive/MyDrive/test/rev6_results/
# (Step-1 and Step-C results are not touched).
#
# HOW TO RUN (T4 GPU):
#   1. Upload this file to /content/drive/MyDrive/test/
#      (AGCMF_Rev5_Step1.py must also be there — this script reuses it)
#   2. %run /content/drive/MyDrive/test/AGCMF_Rev6_StepD_PreReadText_XRV.py
#   Resumable: after a disconnect, run the same line again.
# Approx. time: ~10 min setup + ~70 min training + ~30 min evaluation.
# Output to send back: rev6_results/REVISION_NUMBERS.md
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torchxrayvision'], check=False)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass
import os, glob

STEP1 = '/content/drive/MyDrive/test/AGCMF_Rev5_Step1.py'
if not os.path.exists(STEP1):
    _hits = sorted(glob.glob('/content/drive/MyDrive/**/AGCMF_Rev5_Step1*.py', recursive=True)) + \
            sorted(glob.glob('/content/*AGCMF_Rev5_Step1*.py'))
    if not _hits:
        raise FileNotFoundError(
            'AGCMF_Rev5_Step1.py not found on Drive. Upload it to /content/drive/MyDrive/test/ '
            '(the same file used for Step 1), then re-run this cell.')
    STEP1 = _hits[0]
print(f'Using Step-1 script: {STEP1}')
NEW_TAG = 'rev6_results'
XRV_WEIGHTS = 'densenet121-res224-nih'     # NIH ChestX-ray14 only — do NOT use '-all' (contains MIMIC + CheXpert)

_src = open(STEP1, encoding='utf-8').read()
_i4 = _src.rfind('# ====', 0, _src.index('# CELL 4 / 10'))
print('Running Step-1 cells 1–3 (setup, split, burden, local images)...')
exec(compile(_src[:_i4], STEP1, 'exec'), globals())

import torchxrayvision as xrv
from concurrent.futures import ThreadPoolExecutor

OUT_DIR = f'{BASE_PATH}/{NEW_TAG}'
os.makedirs(OUT_DIR, exist_ok=True)
split_summary.to_csv(f'{OUT_DIR}/split_summary.csv', index=False)
with open(f'{OUT_DIR}/burden_stats.json', 'w') as f:
    json.dump(burden_stats, f, indent=2)

# -----------------------------------------------------------------------------
# D1. Pre-interpretation text: current INDICATION/HISTORY + prior report
# -----------------------------------------------------------------------------
PRE_READ_HEADERS = {'INDICATION', 'HISTORY', 'CLINICAL HISTORY', 'REASON FOR EXAM',
                    'REASON FOR EXAMINATION', 'CLINICAL INFORMATION', 'CLINICAL INDICATION'}
NO_IND, NO_PRIOR = 'No indication available.', 'No prior report available.'


def extract_sections(text, wanted):
    if not text:
        return ''
    heads = [(m.start(), m.end(), m.group(1).strip().upper()) for m in HEADER_RE.finditer(text)]
    parts = []
    for i, (s, e, name) in enumerate(heads):
        if name in wanted:
            end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
            parts.append(text[e:end])
    return re.sub(r'\s+', ' ', ' '.join(parts)).strip()


def prior_summary(text):
    imp = extract_sections(text, {'IMPRESSION'})
    if imp:
        return imp
    fi = extract_sections(text, {'FINDINGS'})
    return fi if fi else (re.sub(r'\s+', ' ', text).strip() if text.strip() else '')


# most recent STRICTLY earlier study of the same patient (uses `studies` from Step-1 Cell 2)
st_sorted = studies.sort_values(['subject_id', 'ts'])
PRIOR_OF = {}
for subj, grp in st_sorted.groupby('subject_id', sort=False):
    ts_list, sid_list = grp['ts'].values, grp['study_id'].values
    for k in range(len(grp)):
        j = k - 1
        while j >= 0 and ts_list[j] >= ts_list[k]:
            j -= 1
        PRIOR_OF[int(sid_list[k])] = int(sid_list[j]) if j >= 0 else None

TEXT_D_PATH = f'{OUT_DIR}/text_cache_preread.pt'
if os.path.exists(TEXT_D_PATH):
    TEXT_CACHE = torch.load(TEXT_D_PATH, weights_only=False)
    _tstats = json.load(open(f'{OUT_DIR}/text_stats.json'))
    print(f'Loaded pre-read text cache ({len(TEXT_CACHE)} studies).')
else:
    keys = [(int(a), int(b)) for a, b in sub_keys.values]
    prior_keys = [(s, PRIOR_OF.get(st)) for s, st in keys]
    print(f'Reading current + prior reports for {len(keys)} studies ({IO_THREADS} threads)...')
    t0 = time.time()
    with ThreadPoolExecutor(IO_THREADS) as ex:
        cur_raw = list(ex.map(lambda kk: read_report(*kk), keys))
        pri_raw = list(ex.map(lambda kk: read_report(kk[0], kk[1]) if kk[1] else '', prior_keys))
    print(f'  done in {time.time() - t0:.0f}s')
    TEXT_CACHE, n_ind, n_prior_study, n_prior_txt, lens = {}, 0, 0, 0, []
    for (subj, stud), (_, pst), cr, pr in zip(keys, prior_keys, cur_raw, pri_raw):
        ind = extract_sections(cr, PRE_READ_HEADERS)
        pri = prior_summary(pr) if pr else ''
        n_ind += int(bool(ind)); n_prior_study += int(pst is not None); n_prior_txt += int(bool(pri))
        enc = tokenizer(ind or NO_IND, pri or NO_PRIOR, padding='max_length', truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors='pt')
        lens.append(int(enc['attention_mask'].sum()))
        TEXT_CACHE[stud] = (enc['input_ids'][0].clone(), enc['attention_mask'][0].clone())
    torch.save(TEXT_CACHE, TEXT_D_PATH)
    _tstats = {'studies': len(keys),
               'with_indication_pct': round(100 * n_ind / len(keys), 1),
               'with_prior_study_pct': round(100 * n_prior_study / len(keys), 1),
               'with_prior_report_text_pct': round(100 * n_prior_txt / len(keys), 1),
               'mean_tokens': round(float(np.mean(lens)), 1),
               'truncated_at_max_len_pct': round(100 * float(np.mean(np.array(lens) >= MAX_TEXT_LEN)), 1)}
    with open(f'{OUT_DIR}/text_stats.json', 'w') as f:
        json.dump(_tstats, f, indent=2)
print('=== Pre-read text (REPORT IN METHODS) ===')
print(json.dumps(_tstats, indent=2))
_ex = next(iter(TEXT_CACHE.values()))
print('Example input:', tokenizer.decode(_ex[0][_ex[1].bool()])[:300])

# -----------------------------------------------------------------------------
# D2. Chest-X-ray-pretrained image encoder (same for M1, M4, M5, M6, M7)
# -----------------------------------------------------------------------------
class XRVEncoder(nn.Module):
    """Takes the Step-1 ImageNet-normalised 3-channel tensor, converts it to the
    1-channel [-1024, 1024] range TorchXRayVision expects, returns 1024-d features."""

    def __init__(self):
        super().__init__()
        self.net = xrv.models.DenseNet(weights=XRV_WEIGHTS)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.std + self.mean                    # back to [0, 1]
        g = x[:, :1].clamp(0, 1)                         # grayscale (channels are identical)
        g = (2.0 * g - 1.0) * 1024.0                     # xrv range
        f = torch.nn.functional.relu(self.net.features(g))
        return torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)   # [B, 1024]


_BaseFusionNet = FusionNet


class XRVFusionNet(_BaseFusionNet):
    def __init__(self, modalities=('img', 'txt', 'clin'), fusion='agcmf'):
        super().__init__(modalities, fusion)
        if 'img' in self.mods:
            self.img_backbone = XRVEncoder()
            self.img_proj = nn.Sequential(nn.Linear(1024, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.GELU())


FusionNet = XRVFusionNet          # every later cell now builds the XRV variant

params_df = pd.DataFrame([{'model': n, 'params_M': round(sum(p.numel() for p in FusionNet(**MODEL_ZOO[n]).parameters()) / 1e6, 2)}
                          for n in TABLE_ORDER])
params_df.to_csv(f'{OUT_DIR}/table9_params.csv', index=False)
print(params_df.to_string(index=False))

env.update({'task': 'pre-read text (current INDICATION/HISTORY + prior report IMPRESSION)',
            'image_encoder': f'TorchXRayVision DenseNet-121 ({XRV_WEIGHTS}, NIH ChestX-ray14 only)',
            'torchxrayvision': getattr(xrv, '__version__', 'unknown')})
with open(f'{OUT_DIR}/environment.json', 'w') as f:
    json.dump(env, f, indent=2)
with open(f'{OUT_DIR}/TASK_DESCRIPTION.md', 'w') as f:
    f.write('# Step D — pre-read text + chest-X-ray-pretrained image encoder\n\n'
            f'Text: {env["task"]}\n\nImage encoder: {env["image_encoder"]}\n\n'
            f'Text stats: {json.dumps(_tstats)}\n')

# -----------------------------------------------------------------------------
# D3. Run Step-1 cells 4–10 unchanged (train M1–M7, Tables 6–8, calibration,
#     missing-modality, MC-Dropout, CheXpert, summary) into rev6_results/
# -----------------------------------------------------------------------------
_rest = _src[_i4:].replace('rev5_results', NEW_TAG)
exec(compile(_rest, STEP1, 'exec'), globals())

with open(f'{OUT_DIR}/REVISION_NUMBERS.md', 'a') as f:
    f.write('\n\n## Step D task\n' + open(f'{OUT_DIR}/TASK_DESCRIPTION.md').read())
print(f'\nSTEP D DONE — send {OUT_DIR}/REVISION_NUMBERS.md')
