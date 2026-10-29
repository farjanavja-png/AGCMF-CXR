# =============================================================================
# AGCMF_Rev5_Step1.py — all Step-1 experiments in one file (Cells 1-10 joined)
#
# HOW TO RUN IN COLAB (T4 GPU runtime):
#   1. Upload this file to /content/drive/MyDrive/test/
#   2. In a Colab cell:   %run /content/drive/MyDrive/test/AGCMF_Rev5_Step1.py
#      (use %run, NOT !python — Drive mounting only works inside the notebook kernel)
#   3. If Colab disconnects, run the same %run line again: finished models and
#      epochs are reloaded from Drive and training resumes where it stopped.
#   4. To train only some models per session, edit RUN_THESE in the CELL 4 section.
#
# Outputs: /content/drive/MyDrive/test/rev5_results/REVISION_NUMBERS.md
#          /content/drive/MyDrive/test/rev5_results.zip
# =============================================================================

# =============================================================================
# =============================================================================
# AGCMF — Scientific Reports Reviewer-1 revision | STEP-1 EXPERIMENTS
# CELL 1 / 10 : SETUP
#
# Runtime > Change runtime type > T4 GPU  (REQUIRED — the last timed run
# started on CPU and never finished).
#
# Run cells 1 -> 10 in order. Every cell is safe to re-run after a Colab
# disconnect: finished models / epochs / results are reloaded from Drive.
# All outputs go to  /content/drive/MyDrive/test/rev5_results/
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'statsmodels'], check=False)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

import os, re, json, time, random, platform, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as T
import torchvision.models as tvm
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import multipletests

# ---------------- paths (same Drive layout as your existing notebook) --------
BASE_PATH      = '/content/drive/MyDrive/test'
IMAGE_DIR      = f'{BASE_PATH}/p10 (1)'
REPORTS_DIR    = f'{BASE_PATH}/p10'
FILTERED_CSV   = f'{BASE_PATH}/available_studies_filtered.csv'
META_CSV       = f'{BASE_PATH}/metadata.csv'            # must contain StudyDate
CHEX_CSV       = f'{BASE_PATH}/mimic-cxr-2.0.0-chexpert.csv'
CHEXPERT_ROOT  = '/content/drive/MyDrive'                # 'Path' column already has CheXpert-v1.0-small/...
CHEXPERT_VALID = f'{CHEXPERT_ROOT}/CheXpert-v1.0-small/valid.csv'
OUT_DIR        = f'{BASE_PATH}/rev5_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- experiment configuration (= Reproducibility statement) -----
BERT_NAME    = 'emilyalsentzer/Bio_ClinicalBERT'
DISEASES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
NUM_CLASSES  = len(DISEASES)
EMBED_DIM    = 256
CLINICAL_DIM = 4
SEED         = 42
BATCH_SIZE   = 16
EPOCHS       = 10
LR_HEAD      = 1e-4      # fusion / projection / classifier layers
LR_BERT      = 2e-5      # Bio_ClinicalBERT backbone
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
MAX_TEXT_LEN = 128
NUM_WORKERS  = 2
USE_AMP      = True      # fp16 mixed precision on T4 (~2x faster). Report it in Methods.
N_BOOT       = 2000      # patient-level cluster bootstrap replicates
MC_PASSES    = 20
AGGREGATE_TO_STUDY = True   # metadata rows are IMAGES; evaluate per STUDY (mean of image logits)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('No GPU found. Runtime > Change runtime type > T4 GPU, then re-run Cell 1.')
USE_AMP = USE_AMP and DEVICE.type == 'cuda'
AMP_DTYPE = torch.float16


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


set_seed()

env = {
    'python': platform.python_version(), 'os': platform.platform(),
    'torch': torch.__version__, 'torchvision': torchvision.__version__,
    'transformers': transformers.__version__,
    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
    'gpu': torch.cuda.get_device_name(0),
    'gpu_mem_GB': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    'seed': SEED, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
    'lr_head': LR_HEAD, 'lr_bert': LR_BERT, 'weight_decay': WEIGHT_DECAY,
    'grad_clip': GRAD_CLIP, 'max_text_len': MAX_TEXT_LEN, 'mixed_precision_fp16': USE_AMP,
    'bootstrap_replicates': N_BOOT, 'mc_dropout_passes': MC_PASSES,
    'evaluation_unit': 'study' if AGGREGATE_TO_STUDY else 'image',
}
with open(f'{OUT_DIR}/environment.json', 'w') as f:
    json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
print('\nCELL 1 done.')


# =============================================================================
# CELL 2 / 10 : DATA — patient-level split, CORRECT leak-free burden score,
#               report-section extraction, text cache, datasets
# =============================================================================

# ---------- 2a. available subset + honest counts ----------
available_df = pd.read_csv(FILTERED_CSV)
missing_cols = [c for c in DISEASES + ['subject_id', 'study_id', 'dicom_id', 'ViewPosition']
                if c not in available_df.columns]
assert not missing_cols, f'Missing columns in {FILTERED_CSV}: {missing_cols}'
available_df['subject_id'] = available_df['subject_id'].astype(int)
available_df['study_id'] = available_df['study_id'].astype(int)

print('=== Available subset (REPORT THESE EXACT NUMBERS IN THE MANUSCRIPT) ===')
print(f"Rows (= images)  : {len(available_df):,}")
print(f"Unique studies   : {available_df['study_id'].nunique():,}")
print(f"Unique patients  : {available_df['subject_id'].nunique():,}")
print(available_df['ViewPosition'].value_counts(dropna=False).to_string())


def patient_level_split(df, test_size=0.15, val_size=0.15, seed=SEED):
    """Identical to the function used for the earlier 0.885 run (same seed -> same split)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df['subject_id']))
    trainval_df, test_df_ = df.iloc[trainval_idx], df.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df['subject_id']))
    return (trainval_df.iloc[train_idx].reset_index(drop=True),
            trainval_df.iloc[val_idx].reset_index(drop=True),
            test_df_.reset_index(drop=True))


train_df, val_df, test_df = patient_level_split(available_df)
p_tr, p_va, p_te = set(train_df.subject_id), set(val_df.subject_id), set(test_df.subject_id)
assert not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te), 'Patient overlap between splits!'

split_summary = pd.DataFrame([
    {'split': n, 'images': len(d), 'studies': d.study_id.nunique(), 'patients': d.subject_id.nunique()}
    for n, d in [('train', train_df), ('val', val_df), ('test', test_df)]])
split_summary.loc[len(split_summary)] = ['total', split_summary.images.sum(),
                                         split_summary.studies.sum(), split_summary.patients.sum()]
print('\n=== Patient-level split (no patient overlap verified) ===')
print(split_summary.to_string(index=False))
split_summary.to_csv(f'{OUT_DIR}/split_summary.csv', index=False)
pd.concat([d.assign(split=n)[['subject_id', 'study_id', 'dicom_id', 'split']]
           for n, d in [('train', train_df), ('val', val_df), ('test', test_df)]]
          ).to_csv(f'{OUT_DIR}/split_assignment.csv', index=False)   # -> upload to GitHub

# ---------- 2b. leak-free history-based disease-burden score ----------
# FIX vs. the old code: the old loop iterated over IMAGE rows, so a study's
# second image saw the first image's (= its own) labels, and the stored value
# was overwritten by the last row -> the current study's labels leaked in.
# Here: one row per STUDY; burden uses only studies with a STRICTLY earlier
# timestamp; studies sharing a timestamp never see each other.
meta = pd.read_csv(META_CSV)
assert 'StudyDate' in meta.columns, 'metadata.csv has no StudyDate column — burden cannot be computed.'
has_time = 'StudyTime' in meta.columns
agg = {'StudyDate': ('StudyDate', 'min')}
if has_time:
    agg['StudyTime'] = ('StudyTime', 'min')
studies = meta.groupby(['subject_id', 'study_id']).agg(**agg).reset_index()
chex = pd.read_csv(CHEX_CSV)
studies = studies.merge(chex, on=['subject_id', 'study_id'], how='inner')
studies['subject_id'] = studies['subject_id'].astype(int)
studies['study_id'] = studies['study_id'].astype(int)

FINDINGS = [d for d in DISEASES if d != 'No Finding']          # 13 findings
studies['pos_count'] = (studies[FINDINGS] == 1).sum(axis=1)
studies['ts'] = studies['StudyDate'].astype(float) * 1e6 + \
    (studies['StudyTime'].fillna(0).astype(float) if has_time else 0.0)

g = (studies.groupby(['subject_id', 'ts'], as_index=False)
            .agg(sum_pos=('pos_count', 'sum'), n=('pos_count', 'size'))
            .sort_values(['subject_id', 'ts']))
g['prior_sum'] = g.groupby('subject_id')['sum_pos'].cumsum() - g['sum_pos']
g['prior_n'] = g.groupby('subject_id')['n'].cumsum() - g['n']
g['burden'] = np.where(g['prior_n'] > 0,
                       g['prior_sum'] / g['prior_n'].clip(lower=1) / len(FINDINGS), 0.0).clip(0, 1)
studies = studies.merge(g[['subject_id', 'ts', 'burden', 'prior_n']], on=['subject_id', 'ts'], how='left')
BURDEN = {(int(r.subject_id), int(r.study_id)): float(r.burden) for r in studies.itertuples()}

# brute-force verification on 300 random studies (proves no current-study leakage)
rng = np.random.default_rng(SEED)
for idx in rng.choice(len(studies), size=min(300, len(studies)), replace=False):
    r = studies.iloc[idx]
    prior = studies[(studies.subject_id == r.subject_id) & (studies.ts < r.ts)]
    expected = 0.0 if len(prior) == 0 else min(1.0, prior.pos_count.mean() / len(FINDINGS))
    assert abs(expected - r.burden) < 1e-9, f'Burden check failed for study {r.study_id}'
assert (studies.loc[studies.prior_n == 0, 'burden'] == 0).all()

sub_keys = available_df[['subject_id', 'study_id']].drop_duplicates()
sub_b = np.array([BURDEN.get((int(a), int(b)), 0.0) for a, b in sub_keys.values])
burden_stats = {
    'studies_with_timestamps': int(len(studies)),
    'studies_with_prior_history_pct_all': round(100 * float((studies.prior_n > 0).mean()), 1),
    'subset_studies': int(len(sub_b)),
    'subset_nonzero_burden_pct': round(100 * float((sub_b > 0).mean()), 1),
    'subset_burden_mean': round(float(sub_b.mean()), 4),
    'corr_burden_vs_CURRENT_pos_count': round(float(np.corrcoef(
        studies.burden, studies.pos_count)[0, 1]), 3),
}
print('\n=== Leak-free burden score (verified by brute force) ===')
print(json.dumps(burden_stats, indent=2))
with open(f'{OUT_DIR}/burden_stats.json', 'w') as f:
    json.dump(burden_stats, f, indent=2)


def clinical_vector(row):
    vp = str(row.get('ViewPosition', '')).upper().strip()
    pa = 1.0 if vp == 'PA' else 0.0
    ap = 1.0 if vp.startswith('AP') else 0.0
    lateral = 1.0 if vp in ('LATERAL', 'LL', 'RL') else 0.0
    burden = BURDEN.get((int(row['subject_id']), int(row['study_id'])), 0.0)
    return [pa, ap, lateral, burden]


# ---------- 2c. FINDINGS + IMPRESSION extraction (as stated in Methods) ----------
HEADER_RE = re.compile(r'^[ \t]*([A-Z][A-Z /&\-]{2,40}):', re.M)
PLACEHOLDER = 'No report available.'


def extract_findings_impression(text):
    if not text or not text.strip():
        return PLACEHOLDER
    heads = [(m.start(), m.end(), m.group(1).strip().upper()) for m in HEADER_RE.finditer(text)]
    parts = []
    for i, (s, e, name) in enumerate(heads):
        if name in ('FINDINGS', 'IMPRESSION'):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
            parts.append(text[e:end])
    out = ' '.join(parts) if parts else text
    out = re.sub(r'\s+', ' ', out).strip()
    return out if out else PLACEHOLDER


def read_report(subject_id, study_id):
    sid, stid = f'p{subject_id}', f's{study_id}'
    for p in [f'{REPORTS_DIR}/{sid}/{stid}.txt', f'{REPORTS_DIR}/{sid} (1)/{stid}.txt']:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except OSError:
            continue
    return ''


from concurrent.futures import ThreadPoolExecutor
IO_THREADS = 32          # Drive is slow per file but fine in parallel

tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
TEXT_CACHE_PATH = f'{OUT_DIR}/text_cache.pt'
if os.path.exists(TEXT_CACHE_PATH):
    TEXT_CACHE = torch.load(TEXT_CACHE_PATH, weights_only=False)
    print(f'\nLoaded tokenised report cache ({len(TEXT_CACHE)} studies).')
else:
    print(f'\nReading {len(sub_keys)} reports from Drive in parallel ({IO_THREADS} threads)...')
    keys = [(int(a), int(b)) for a, b in sub_keys.values]
    raws, t0 = {}, time.time()
    with ThreadPoolExecutor(IO_THREADS) as ex:
        for k, (key, raw) in enumerate(zip(keys, ex.map(lambda kk: read_report(*kk), keys))):
            raws[key[1]] = raw
            if (k + 1) % 500 == 0 or k + 1 == len(keys):
                print(f'  {k + 1}/{len(keys)} reports read  ({time.time() - t0:.0f}s)')
    TEXT_CACHE, n_missing, n_sections = {}, 0, 0
    for stud, raw in raws.items():
        n_missing += int(raw.strip() == '')
        n_sections += int(bool(re.search(r'(FINDINGS|IMPRESSION)\s*:', raw)))
        enc = tokenizer(extract_findings_impression(raw), padding='max_length', truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors='pt')
        TEXT_CACHE[stud] = (enc['input_ids'][0].clone(), enc['attention_mask'][0].clone())
    torch.save(TEXT_CACHE, TEXT_CACHE_PATH)
    print(f'Reports: {len(TEXT_CACHE)} studies | missing report file: {n_missing} | '
          f'with FINDINGS/IMPRESSION header: {n_sections}')
    if n_missing > 0.05 * len(TEXT_CACHE):
        print('  WARNING: >5% of reports missing — check REPORTS_DIR before training!')

# ---------- 2d. transforms (exactly the augmentations listed in Methods) ----------
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ToTensor(), NORM])
tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])


def find_image_on_drive(row):
    sid, stid, did = f"p{row['subject_id']}", f"s{row['study_id']}", row['dicom_id']
    for p in [f'{IMAGE_DIR}/{sid} (1)/{stid}/{did}.jpg', f'{IMAGE_DIR}/{sid}/{stid}/{did}.jpg']:
        if os.path.exists(p):
            return p
    return None


# ---------- 2e. copy every image ONCE to Colab's local disk, pre-resized ----------
# Saved as lossless 224x224 PNG with the same bilinear resize torchvision applies,
# so results are identical to resizing on the fly — but each epoch reads local disk
# instead of Drive (much faster). Resumable: already-copied images are skipped.
# (Local disk is wiped when the Colab runtime is reset; this step then re-runs.)
LOCAL_IMG_DIR = '/content/local_images_224'
os.makedirs(LOCAL_IMG_DIR, exist_ok=True)


def local_path(row):
    return f"{LOCAL_IMG_DIR}/{row['dicom_id']}.png"


def copy_one(row):
    dst = local_path(row)
    if os.path.exists(dst):
        return 'ok'
    src = find_image_on_drive(row)
    if src is None:
        return 'missing'
    try:
        Image.open(src).convert('RGB').resize((224, 224), Image.BILINEAR).save(dst)
        return 'ok'
    except Exception:
        return 'error'


rows_all = [r for _, r in available_df.iterrows()]
print(f'\nCopying {len(rows_all)} images to local disk ({IO_THREADS} threads, one-time)...')
status, t0 = {'ok': 0, 'missing': 0, 'error': 0}, time.time()
with ThreadPoolExecutor(IO_THREADS) as ex:
    for k, st in enumerate(ex.map(copy_one, rows_all)):
        status[st] += 1
        if (k + 1) % 500 == 0 or k + 1 == len(rows_all):
            print(f'  {k + 1}/{len(rows_all)} images  {status}  ({time.time() - t0:.0f}s)')
if status['missing'] or status['error']:
    print('  WARNING: some images missing/unreadable — they will be zero images. Report this count.')


def find_image(row):
    p = local_path(row)
    return p if os.path.exists(p) else None


def label_vector(row):
    # CheXpert convention used throughout: uncertain (-1) and blank -> 0 (U-Zeros)
    return [0.0 if pd.isna(row.get(c, np.nan)) or row.get(c, 0) == -1 else float(row.get(c, 0))
            for c in DISEASES]


class TrimodalDataset(Dataset):
    def __init__(self, df, transform, need_image=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.need_image = need_image
        self.clin = torch.tensor([clinical_vector(r) for _, r in self.df.iterrows()], dtype=torch.float32)
        self.lab = torch.tensor([label_vector(r) for _, r in self.df.iterrows()], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = torch.zeros(3, 224, 224)
        if self.need_image:
            p = find_image(row)
            try:
                img = self.transform(Image.open(p).convert('RGB'))
            except Exception:
                pass
        ids, mask = TEXT_CACHE.get(int(row['study_id']),
                                   (torch.zeros(MAX_TEXT_LEN, dtype=torch.long),
                                    torch.zeros(MAX_TEXT_LEN, dtype=torch.long)))
        return img, ids, mask, self.clin[i], self.lab[i]


n_bad = sum(find_image(r) is None for _, r in test_df.iterrows())
print(f'\nImage check: {n_bad}/{len(test_df)} test images not available locally (should be 0).')
print('\nCELL 2 done.')


# =============================================================================
# CELL 3 / 10 : MODELS  M1–M7  (all share the SAME encoders / head / training)
#   M1 image | M2 text | M3 clinical | M4 image+clinical | M5 image+text
#   M6 trimodal BiLSTM fusion | M7 AGCMF (identical computation to your model)
# =============================================================================
class FusionNet(nn.Module):
    def __init__(self, modalities=('img', 'txt', 'clin'), fusion='agcmf'):
        super().__init__()
        self.mods, self.fusion, D = tuple(modalities), fusion, EMBED_DIM
        if 'img' in self.mods:
            resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            self.img_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.img_proj = nn.Sequential(nn.Linear(2048, D), nn.LayerNorm(D), nn.GELU())
        if 'txt' in self.mods:
            self.bert = AutoModel.from_pretrained(BERT_NAME)
            self.txt_proj = nn.Sequential(nn.Linear(self.bert.config.hidden_size, D), nn.LayerNorm(D), nn.GELU())
        if 'clin' in self.mods:
            self.clin_proj = nn.Sequential(nn.Linear(CLINICAL_DIM, D), nn.LayerNorm(D), nn.GELU())

        if fusion == 'agcmf':
            assert self.mods == ('img', 'txt', 'clin')
            self.cross_attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True, dropout=0.1)
            self.self_attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True, dropout=0.1)
            self.gate = nn.Sequential(nn.Linear(D * 3, D), nn.Sigmoid())
        elif fusion == 'bilstm':
            self.lstm = nn.LSTM(D, D // 2, batch_first=True, bidirectional=True)
        elif fusion == 'concat':
            self.fuse = (nn.Sequential(nn.Linear(D * len(self.mods), D), nn.LayerNorm(D), nn.GELU())
                         if len(self.mods) > 1 else nn.Identity())
        else:
            raise ValueError(fusion)
        self.classifier = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, NUM_CLASSES))

    def forward(self, image, input_ids, attn_mask, clinical,
                zero_image=False, zero_text=False, zero_clinical=False, return_gate=False):
        B, dev = clinical.size(0), clinical.device
        zeros = lambda: torch.zeros(B, EMBED_DIM, device=dev)
        tok = {}
        if 'img' in self.mods:
            tok['img'] = zeros() if zero_image else self.img_proj(self.img_backbone(image).flatten(1))
        if 'txt' in self.mods:
            if zero_text:
                tok['txt'] = zeros()
            else:
                cls = self.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
                tok['txt'] = self.txt_proj(cls)
        if 'clin' in self.mods:
            tok['clin'] = zeros() if zero_clinical else self.clin_proj(clinical)
        toks = [tok[m].float() for m in self.mods]

        gate = None
        if self.fusion == 'agcmf':
            h_i = toks[0]
            tokens = torch.stack(toks, dim=1)
            cross_out, _ = self.cross_attn(tokens, tokens, tokens)
            refined, _ = self.self_attn(cross_out, cross_out, cross_out)
            pooled_tokens, pooled_refined = tokens.mean(dim=1), refined.mean(dim=1)
            gate = self.gate(torch.cat([pooled_tokens, pooled_refined, h_i], dim=1))
            fused = gate * pooled_tokens + (1 - gate) * pooled_refined
        elif self.fusion == 'bilstm':
            out, _ = self.lstm(torch.stack(toks, dim=1))
            fused = out.mean(dim=1)
        else:
            fused = self.fuse(torch.cat(toks, dim=1))
        logits = self.classifier(fused)
        return (logits, gate) if return_gate else logits


MODEL_ZOO = {
    'M7_AGCMF':        dict(modalities=('img', 'txt', 'clin'), fusion='agcmf'),
    'M1_Image':        dict(modalities=('img',),               fusion='concat'),
    'M5_Img+Text':     dict(modalities=('img', 'txt'),         fusion='concat'),
    'M6_BiLSTM':       dict(modalities=('img', 'txt', 'clin'), fusion='bilstm'),
    'M4_Img+Clin':     dict(modalities=('img', 'clin'),        fusion='concat'),
    'M2_Text':         dict(modalities=('txt',),               fusion='concat'),
    'M3_Clinical':     dict(modalities=('clin',),              fusion='concat'),
}
TABLE_ORDER = ['M1_Image', 'M2_Text', 'M3_Clinical', 'M4_Img+Clin', 'M5_Img+Text', 'M6_BiLSTM', 'M7_AGCMF']
MOD_LABEL = {'M1_Image': 'Image only', 'M2_Text': 'Text only', 'M3_Clinical': 'Clinical only',
             'M4_Img+Clin': 'Image + Clinical', 'M5_Img+Text': 'Image + Text',
             'M6_BiLSTM': 'Image + Text + Clinical (BiLSTM)', 'M7_AGCMF': 'Image + Text + Clinical (AGCMF)'}

# parameter counts (for Table 9) — built on CPU, then freed
param_rows = []
for name in TABLE_ORDER:
    m = FusionNet(**MODEL_ZOO[name])
    param_rows.append({'model': name, 'params_M': round(sum(p.numel() for p in m.parameters()) / 1e6, 2)})
    del m
params_df = pd.DataFrame(param_rows)
params_df.to_csv(f'{OUT_DIR}/table9_params.csv', index=False)
print(params_df.to_string(index=False))
print('\nCELL 3 done.')


# =============================================================================
# CELL 4 / 10 : TRAIN M1–M7 on the corrected 8,502-row subset
#   - same seed / epochs / optimiser / augmentation for every model
#   - best checkpoint chosen on VALIDATION mean AUC only (test never touched)
#   - resumable: a disconnect resumes from the last finished epoch
#   - if Colab time is short, set RUN_THESE = ['M7_AGCMF'] etc. and run in pieces
# =============================================================================
RUN_THESE = list(MODEL_ZOO)        # order: M7, M1, M5, M6, M4, M2, M3


def make_loader(df, train, need_image, generator=None):
    ds = TrimodalDataset(df, tf_train if train else tf_eval, need_image=need_image)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=NUM_WORKERS,
                      pin_memory=True, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def to_dev(batch):
    img, ids, mask, clin, y = batch
    return (img.to(DEVICE, non_blocking=True), ids.to(DEVICE, non_blocking=True),
            mask.to(DEVICE, non_blocking=True), clin.to(DEVICE, non_blocking=True),
            y.to(DEVICE, non_blocking=True))


@torch.no_grad()
def predict_logits(model, loader, **zero_flags):
    model.eval()
    Ls, Ys = [], []
    for batch in loader:
        img, ids, mask, clin, y = to_dev(batch)
        with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
            out = model(img, ids, mask, clin, **zero_flags)
        Ls.append(out.float().cpu().numpy()); Ys.append(y.cpu().numpy())
    return np.vstack(Ys), np.vstack(Ls)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fast_auc(y, p):
    """Mann–Whitney AUC (identical to sklearn roc_auc_score, handles ties)."""
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_class_auc(Y, P):
    return np.array([fast_auc(Y[:, i], P[:, i]) for i in range(Y.shape[1])])


def mean_auc(Y, P):
    return float(np.nanmean(per_class_auc(Y, P)))


def compute_pos_weight(labels, min_w=1.0, max_w=10.0):
    pos = labels.sum(axis=0); neg = labels.shape[0] - pos
    return torch.tensor(np.clip(neg / np.clip(pos, 1e-6, None), min_w, max_w), dtype=torch.float32)


def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=USE_AMP)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def train_one(name):
    cfg = MODEL_ZOO[name]
    pred_path = f'{OUT_DIR}/{name}_predictions.npz'
    if os.path.exists(pred_path):
        print(f'[{name}] already finished — skipping (delete {pred_path} to retrain).')
        return
    print(f'\n================ {name}  {cfg} ================')
    set_seed(SEED)
    model = FusionNet(**cfg).to(DEVICE)
    need_img = 'img' in cfg['modalities']
    gen = torch.Generator(); gen.manual_seed(SEED)
    tl = make_loader(train_df, True, need_img, generator=gen)
    vl = make_loader(val_df, False, need_img)
    te = make_loader(test_df, False, need_img)

    pos_weight = compute_pos_weight(tl.dataset.lab.numpy()).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bert_p = [p for n, p in model.named_parameters() if n.startswith('bert.')]
    other_p = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
    groups = [{'params': other_p, 'lr': LR_HEAD}] + ([{'params': bert_p, 'lr': LR_BERT}] if bert_p else [])
    optimizer = optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()

    best_ckpt, last_ckpt = f'{OUT_DIR}/{name}_best.pt', f'{OUT_DIR}/{name}_last.pt'
    start_ep, best_auc, best_ep, log = 0, -1.0, -1, []
    if os.path.exists(last_ckpt):
        st = torch.load(last_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st['model']); optimizer.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler'])
        start_ep, best_auc, best_ep, log = st['epoch'] + 1, st['best_auc'], st['best_ep'], st['log']
        print(f'Resuming {name} from epoch {start_ep + 1}')

    for ep in range(start_ep, EPOCHS):
        gen.manual_seed(SEED + ep)            # deterministic, resumable shuffling/augmentation
        set_seed(SEED + ep)
        model.train(); t0 = time.time(); run_loss = 0.0
        for batch in tl:
            img, ids, mask, clin, y = to_dev(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(img, ids, mask, clin)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item()
        Yv, Lv = predict_logits(model, vl)
        v_auc = mean_auc(Yv, sigmoid(Lv))
        sec = time.time() - t0
        log.append({'epoch': ep + 1, 'train_loss': run_loss / len(tl), 'val_mean_auc': v_auc, 'seconds': sec})
        flag = ''
        if v_auc > best_auc:
            best_auc, best_ep, flag = v_auc, ep + 1, '  <- best'
            torch.save(model.state_dict(), best_ckpt)
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': ep, 'best_auc': best_auc, 'best_ep': best_ep, 'log': log}, last_ckpt)
        print(f'[{name}] epoch {ep + 1}/{EPOCHS}  loss={run_loss / len(tl):.4f}  '
              f'val_mean_AUC={v_auc:.4f}  ({sec / 60:.1f} min){flag}')

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    Yv, Lv = predict_logits(model, vl)
    Yt, Lt = predict_logits(model, te)
    np.savez(pred_path, val_Y=Yv, val_logits=Lv, test_Y=Yt, test_logits=Lt)
    pd.DataFrame(log).to_csv(f'{OUT_DIR}/{name}_train_log.csv', index=False)
    info = {'model': name, 'best_epoch': best_ep, 'best_val_mean_auc': best_auc,
            'train_minutes_total': round(sum(r['seconds'] for r in log) / 60, 1),
            'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    with open(f'{OUT_DIR}/{name}_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f'[{name}] DONE  {info}')
    del model, optimizer; torch.cuda.empty_cache()


for _name in RUN_THESE:
    train_one(_name)

done = [n for n in TABLE_ORDER if os.path.exists(f'{OUT_DIR}/{n}_predictions.npz')]
print(f'\nFinished models: {done}')
print(f'Still to train : {[n for n in TABLE_ORDER if n not in done]}')
print('\nCELL 4 done.')


# =============================================================================
# CELL 5 / 10 : NEW TABLE 6 (ablation) + TABLE 7 (per-disease AUC)
#   - evaluation unit = STUDY (image logits averaged within a study)
#   - 95% CIs by PATIENT-level cluster bootstrap (2,000 replicates)
#   - per-class thresholds chosen on VALIDATION (Youden J), applied to test
# =============================================================================
def aggregate(Y, L, df):
    """Average logits over images of the same study. Returns Y, logits, patient ids."""
    if not AGGREGATE_TO_STUDY:
        return Y, L, df['subject_id'].values
    sid = df['study_id'].values
    Lg = pd.DataFrame(L).groupby(sid, sort=True).mean()
    Yg = pd.DataFrame(Y).groupby(sid, sort=True).max()
    pat = pd.Series(df['subject_id'].values, index=sid).groupby(level=0).first().loc[Lg.index]
    return Yg.values, Lg.values, pat.values


def load_model_preds(name):
    d = np.load(f'{OUT_DIR}/{name}_predictions.npz')
    vY, vL, vpat = aggregate(d['val_Y'], d['val_logits'], val_df)
    tY, tL, tpat = aggregate(d['test_Y'], d['test_logits'], test_df)
    return dict(vY=vY, vL=vL, vP=sigmoid(vL), vpat=vpat, tY=tY, tL=tL, tP=sigmoid(tL), tpat=tpat)


def youden_thresholds(Y, P):
    thr = np.full(Y.shape[1], 0.5)
    for i in range(Y.shape[1]):
        if 0 < Y[:, i].sum() < len(Y):
            fpr, tpr, th = roc_curve(Y[:, i], P[:, i])
            th = np.where(np.isfinite(th), th, 1.0)
            thr[i] = th[np.argmax(tpr - fpr)]
    return thr


def threshold_metrics(Y, P, thr):
    pred = (P >= thr).astype(int)
    tp = ((pred == 1) & (Y == 1)).sum(0); tn = ((pred == 0) & (Y == 0)).sum(0)
    fp = ((pred == 1) & (Y == 0)).sum(0); fn = ((pred == 0) & (Y == 1)).sum(0)
    valid = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = tp / (tp + fn); spec = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn); acc = (tp + tn) / len(Y)
        ppv = tp / (tp + fp); npv = tn / (tn + fn)
    m = lambda a: float(np.nanmean(np.where(valid, a, np.nan)))
    return {'macro_F1': m(f1), 'sensitivity': m(sens), 'specificity': m(spec),
            'accuracy': m(acc), 'PPV': m(ppv), 'NPV': m(npv)}


def summary_metrics(Y, P, thr):
    out = {'mean_AUC': mean_auc(Y, P)}
    out.update(threshold_metrics(Y, P, thr))
    return out


def cluster_bootstrap(stat_fn, patients, n_boot=N_BOOT, seed=SEED):
    """stat_fn(idx) -> float or dict; resamples PATIENTS with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    groups = {p: np.where(patients == p)[0] for p in uniq}
    reps = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        reps.append(stat_fn(idx))
    return reps


def ci(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


fmt = lambda v, lo, hi: f'{v:.3f} ({lo:.3f}–{hi:.3f})'

AVAILABLE = [n for n in TABLE_ORDER if os.path.exists(f'{OUT_DIR}/{n}_predictions.npz')]
missing = [n for n in TABLE_ORDER if n not in AVAILABLE]
if missing:
    print(f'WARNING — not trained yet (re-run Cell 4): {missing}\n')

PRED, THR, SUMM = {}, {}, {}
rows6 = []
for name in AVAILABLE:
    d = load_model_preds(name); PRED[name] = d
    THR[name] = youden_thresholds(d['vY'], d['vP'])
    s = summary_metrics(d['tY'], d['tP'], THR[name]); SUMM[name] = s
    boots = cluster_bootstrap(lambda idx: summary_metrics(d['tY'][idx], d['tP'][idx], THR[name]), d['tpat'])
    row = {'Model': name, 'Modality': MOD_LABEL[name]}
    for k in ['mean_AUC', 'macro_F1', 'sensitivity', 'specificity', 'accuracy']:
        lo, hi = ci([b[k] for b in boots])
        row[k] = round(s[k], 3); row[f'{k}_CI'] = f'{lo:.3f}–{hi:.3f}'
    rows6.append(row)

table6 = pd.DataFrame(rows6)
if 'M1_Image' in SUMM:
    table6['dAUC_vs_M1_pp'] = [round(100 * (SUMM[n]['mean_AUC'] - SUMM['M1_Image']['mean_AUC']), 1)
                               for n in table6.Model]
table6.to_csv(f'{OUT_DIR}/table6_corrected_ablation.csv', index=False)

ref = AVAILABLE[0]
n_test_units = len(PRED[ref]['tY'])
print(f"=== NEW TABLE 6 — corrected subset, test set n = {n_test_units} "
      f"{'studies' if AGGREGATE_TO_STUDY else 'images'}, "
      f"{len(np.unique(PRED[ref]['tpat']))} patients ===")
print(table6[['Model', 'Modality', 'mean_AUC', 'mean_AUC_CI', 'macro_F1', 'sensitivity',
              'specificity'] + (['dAUC_vs_M1_pp'] if 'dAUC_vs_M1_pp' in table6 else [])].to_string(index=False))

# ---------- Table 7: per-disease AUC ----------
t7 = pd.DataFrame({'Disease': DISEASES,
                   'test_positives': PRED[ref]['tY'].sum(0).astype(int)})
for name in AVAILABLE:
    t7[name] = np.round(per_class_auc(PRED[name]['tY'], PRED[name]['tP']), 3)
t7.to_csv(f'{OUT_DIR}/table7_corrected_per_disease_auc.csv', index=False)
print('\n=== NEW TABLE 7 — per-disease AUC ===')
print(t7.to_string(index=False))

if 'M7_AGCMF' in SUMM:
    best_cols = t7[AVAILABLE].idxmax(axis=1)
    print(f"\nClasses where M7 has the highest AUC: {(best_cols == 'M7_AGCMF').sum()} / {NUM_CLASSES}"
          "   <- report this honestly (do NOT write 'best for all classes' unless it is 14/14)")
print('\nCELL 5 done.')


# =============================================================================
# CELL 6 / 10 : NEW TABLE 8 — statistical comparison M7 vs M1–M6
#   (a) per-class DeLong test, Holm-corrected across the 14 classes
#   (b) paired patient-level bootstrap of the mean-AUC difference,
#       Holm-corrected across the 6 baselines   <- primary test
#   (c) Wilcoxon on 14 paired per-class AUCs (supplementary, for continuity)
# =============================================================================
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    Tm = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        Tm[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = Tm
    return T2


def delong_roc_test(y_true, prob_a, prob_b):
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind='mergesort')
    m = int(y_true.sum()); n = len(y_true) - m
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    preds = np.vstack([prob_a[order], prob_b[order]])
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_compute_midrank(r) for r in pos])
    ty = np.array([_compute_midrank(r) for r in neg])
    tz = np.array([_compute_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


REF = 'M7_AGCMF'
assert REF in PRED, 'Train M7_AGCMF first (Cell 4).'
d7 = PRED[REF]
rows8, delong_long = [], []
for base in [n for n in TABLE_ORDER if n != REF and n in PRED]:
    db = PRED[base]
    assert np.array_equal(d7['tY'], db['tY']), 'Test sets differ — cannot pair.'
    # (a) DeLong per class
    ps = []
    for i, dis in enumerate(DISEASES):
        a7, ab, p = delong_roc_test(d7['tY'][:, i], d7['tP'][:, i], db['tP'][:, i])
        ps.append(p)
        delong_long.append({'comparison': f'M7 vs {base}', 'disease': dis,
                            'AUC_M7': a7, 'AUC_base': ab, 'p_raw': p})
    ps = np.array(ps); ok = ~np.isnan(ps)
    p_holm = np.full_like(ps, np.nan); p_holm[ok] = multipletests(ps[ok], method='holm')[1]
    for k, r in enumerate(delong_long[-NUM_CLASSES:]):
        r['p_holm'] = p_holm[k]
    # (b) paired cluster bootstrap of delta mean AUC
    obs = mean_auc(d7['tY'], d7['tP']) - mean_auc(db['tY'], db['tP'])
    deltas = np.array(cluster_bootstrap(
        lambda idx: mean_auc(d7['tY'][idx], d7['tP'][idx]) - mean_auc(db['tY'][idx], db['tP'][idx]),
        d7['tpat']))
    p_boot = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    p_boot = max(p_boot, 1.0 / N_BOOT)        # resolution limit -> reported as "< 1/N_BOOT"
    lo, hi = ci(deltas)
    # (c) Wilcoxon on per-class AUCs
    a7c, abc = per_class_auc(d7['tY'], d7['tP']), per_class_auc(db['tY'], db['tP'])
    okc = ~np.isnan(a7c) & ~np.isnan(abc)
    w_stat, w_p = stats.wilcoxon(a7c[okc], abc[okc])
    rows8.append({'Comparison': f'M7 vs {base}', 'AUC_M7': round(SUMM[REF]['mean_AUC'], 3),
                  'AUC_base': round(SUMM[base]['mean_AUC'], 3), 'dAUC_pp': round(100 * obs, 1),
                  'dAUC_95CI_pp': f'{100 * lo:.1f} to {100 * hi:.1f}', 'p_bootstrap': p_boot,
                  'DeLong_sig_classes_Holm': f'{int((p_holm[ok] < 0.05).sum())}/{int(ok.sum())}',
                  'Wilcoxon_W': float(w_stat), 'Wilcoxon_p': float(w_p)})

table8 = pd.DataFrame(rows8)
if len(table8):
    table8['p_bootstrap_Holm'] = multipletests(table8['p_bootstrap'], method='holm')[1]
    table8['significant_after_Holm'] = table8['p_bootstrap_Holm'] < 0.05
table8.to_csv(f'{OUT_DIR}/table8_corrected_significance.csv', index=False)
pd.DataFrame(delong_long).to_csv(f'{OUT_DIR}/table8b_delong_per_class.csv', index=False)
print(f'=== NEW TABLE 8 (p_bootstrap of {1 / N_BOOT:.4f} means p < {1 / N_BOOT:.4f}) ===')
print(table8.to_string(index=False))
print('\nIf M7 vs M5 (Image+Text) is NOT significant, say so in the paper — it is consistent with the')
print('reviewer\'s point that text carries most of the signal.')
print('\nCELL 6 done.')


# =============================================================================
# CELL 7 / 10 : CALIBRATION of M7 (goes into main Results, Reviewer comment 12)
#   raw vs temperature scaling vs Platt scaling — all FITTED ON VALIDATION,
#   evaluated on test. AUC is unchanged by these monotone per-class maps.
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ece_score(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _logreg(x, y):
    try:
        lr = LogisticRegression(penalty=None, max_iter=5000)
    except TypeError:
        lr = LogisticRegression(C=1e9, max_iter=5000)
    lr.fit(x.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calib_slope_intercept(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return _logreg(np.log(p / (1 - p)), y)          # same definition as the earlier run


def fit_temperature(y, logit):
    def nll(logT):
        q = np.clip(sigmoid(logit / np.exp(logT)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x))


vY, vL, tY, tL = d7['vY'], d7['vL'], d7['tY'], d7['tL']
P_raw, P_temp, P_platt = sigmoid(tL), np.zeros_like(tL), np.zeros_like(tL)
temps, platt_params = [], []
for i in range(NUM_CLASSES):
    Tc = fit_temperature(vY[:, i], vL[:, i]); temps.append(Tc)
    P_temp[:, i] = sigmoid(tL[:, i] / Tc)
    if 0 < vY[:, i].sum() < len(vY):
        a, b = _logreg(vL[:, i], vY[:, i])
    else:
        a, b = 1.0, 0.0
    platt_params.append((a, b))
    P_platt[:, i] = sigmoid(a * tL[:, i] + b)

cal_rows = []
for i, dis in enumerate(DISEASES):
    y = tY[:, i]
    if not (0 < y.sum() < len(y)):
        continue
    r = {'disease': dis, 'positives': int(y.sum()), 'temperature': round(temps[i], 3)}
    for tag, P in [('raw', P_raw), ('temp', P_temp), ('platt', P_platt)]:
        p = P[:, i]; s, c = calib_slope_intercept(y, p)
        r[f'brier_{tag}'] = float(np.mean((p - y) ** 2)); r[f'ece_{tag}'] = ece_score(y, p)
        r[f'slope_{tag}'] = s; r[f'intercept_{tag}'] = c
    cal_rows.append(r)
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(f'{OUT_DIR}/calibration_per_class.csv', index=False)

macro = {tag: {k: float(cal_df[f'{k}_{tag}'].mean()) for k in ['brier', 'ece', 'slope', 'intercept']}
         for tag in ['raw', 'temp', 'platt']}
# patient-level CI for macro Brier (raw and Platt)
valid_i = [DISEASES.index(d) for d in cal_df.disease]
for tag, P in [('raw', P_raw), ('platt', P_platt)]:
    b = cluster_bootstrap(lambda idx: float(np.mean([np.mean((P[idx, i] - tY[idx, i]) ** 2) for i in valid_i])),
                          d7['tpat'], n_boot=1000)
    macro[tag]['brier_95CI'] = '%.4f–%.4f' % ci(b)
with open(f'{OUT_DIR}/calibration_macro.json', 'w') as f:
    json.dump(macro, f, indent=2)
print('=== M7 calibration on test (macro over classes) ===')
print(pd.DataFrame(macro).T.to_string())

# reliability diagrams: raw vs Platt, per class
fig, axes = plt.subplots(4, 4, figsize=(14, 14)); axes = axes.ravel()
for k, i in enumerate(valid_i):
    ax = axes[k]
    for P, lab in [(P_raw, 'raw'), (P_platt, 'Platt')]:
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for j in range(10):
            m = (P[:, i] >= edges[j]) & (P[:, i] <= edges[j + 1])
            if m.sum() >= 5:
                xs.append(P[m, i].mean()); ys.append(tY[m, i].mean())
        ax.plot(xs, ys, 'o-', label=lab, ms=3)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8); ax.set_title(DISEASES[i], fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if k == 0:
        ax.legend(fontsize=8)
for ax in axes[len(valid_i):]:
    ax.axis('off')
fig.supxlabel('Mean predicted probability'); fig.supylabel('Observed frequency')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/fig_reliability_M7.png', dpi=200); plt.close()
print(f'Saved reliability diagram -> {OUT_DIR}/fig_reliability_M7.png')
print('\nCELL 7 done.')


# =============================================================================
# CELL 8 / 10 : M7 inference-time missing-modality analysis + MC-Dropout
#   (the trained M1–M6 in Table 6 are the proper ablation; this cell answers
#    "how does the SAME deployed M7 degrade when a modality is absent")
# =============================================================================
m7 = FusionNet(**MODEL_ZOO['M7_AGCMF']).to(DEVICE)
m7.load_state_dict(torch.load(f'{OUT_DIR}/M7_AGCMF_best.pt', map_location=DEVICE))
test_loader_img = make_loader(test_df, False, need_image=True)

SCENARIOS = {
    'Full trimodal':     dict(zero_image=False, zero_text=False, zero_clinical=False),
    'Image + Text':      dict(zero_image=False, zero_text=False, zero_clinical=True),
    'Text + Clinical':   dict(zero_image=True,  zero_text=False, zero_clinical=False),
    'Image + Clinical':  dict(zero_image=False, zero_text=True,  zero_clinical=False),
    'Text only':         dict(zero_image=True,  zero_text=False, zero_clinical=True),
    'Image only':        dict(zero_image=False, zero_text=True,  zero_clinical=True),
    'Clinical only':     dict(zero_image=True,  zero_text=True,  zero_clinical=False),
}
mm_rows = []
for sc, flags in SCENARIOS.items():
    Y, L = predict_logits(m7, test_loader_img, **flags)
    Yg, Lg, pat = aggregate(Y, L, test_df); Pg = sigmoid(Lg)
    auc = mean_auc(Yg, Pg)
    lo, hi = ci(cluster_bootstrap(lambda idx: mean_auc(Yg[idx], Pg[idx]), pat, n_boot=1000))
    f1 = threshold_metrics(Yg, Pg, THR['M7_AGCMF'])['macro_F1']
    mm_rows.append({'scenario': sc, 'mean_AUC': round(auc, 3), 'AUC_95CI': f'{lo:.3f}–{hi:.3f}',
                    'macro_F1': round(f1, 3)})
    print(f'{sc:18s} mean_AUC={auc:.3f} ({lo:.3f}–{hi:.3f})  macro_F1={f1:.3f}')
mm_df = pd.DataFrame(mm_rows); mm_df.to_csv(f'{OUT_DIR}/missing_modality_M7.csv', index=False)


# ---------- MC-Dropout ----------
def enable_mc_dropout(model):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.MultiheadAttention)):
            mod.train()


set_seed(SEED)
enable_mc_dropout(m7)
passes = []
with torch.no_grad():
    for k in range(MC_PASSES):
        Ls = []
        for batch in test_loader_img:
            img, ids, mask, clin, y = to_dev(batch)
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                Ls.append(m7(img, ids, mask, clin).float().cpu().numpy())
        _, Lg, _ = aggregate(np.zeros((len(test_df), NUM_CLASSES)), np.vstack(Ls), test_df)
        passes.append(sigmoid(Lg))
        print(f'  MC pass {k + 1}/{MC_PASSES}')
passes = np.stack(passes)                     # [passes, studies, classes]
mc_mean, mc_std = passes.mean(0), passes.std(0)
m7.eval()

# does uncertainty flag errors? AUROC of predictive SD for detecting misclassified cases
err = (mc_mean >= THR['M7_AGCMF']).astype(int) != d7['tY']
mc_rows = []
for i, dis in enumerate(DISEASES):
    e = err[:, i].astype(int)
    mc_rows.append({'disease': dis, 'mean_predictive_SD': round(float(mc_std[:, i].mean()), 4),
                    'AUROC_SD_detects_error': round(fast_auc(e, mc_std[:, i]), 3)
                    if 0 < e.sum() < len(e) else np.nan})
mc_df = pd.DataFrame(mc_rows); mc_df.to_csv(f'{OUT_DIR}/mc_dropout_M7.csv', index=False)
print('\n=== MC-Dropout (20 passes) ===')
print(mc_df.to_string(index=False))
print(f"SD range: {mc_df.mean_predictive_SD.min():.3f}–{mc_df.mean_predictive_SD.max():.3f}")

# ---------- gate statistics (for Fig. 5 text) ----------
gates = []
with torch.no_grad():
    for batch in test_loader_img:
        img, ids, mask, clin, y = to_dev(batch)
        with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
            _, gt = m7(img, ids, mask, clin, return_gate=True)
        gates.append(gt.float().mean(1).cpu().numpy())
gates = np.concatenate(gates)
gate_stats = {'gate_mean': float(gates.mean()), 'gate_sd': float(gates.std()),
              'gate_p5': float(np.percentile(gates, 5)), 'gate_p95': float(np.percentile(gates, 95))}
with open(f'{OUT_DIR}/gate_stats_M7.json', 'w') as f:
    json.dump(gate_stats, f, indent=2)
print('\nPer-sample gate (weight on raw token mean vs attention-refined):', gate_stats)
print('\nCELL 8 done.')


# =============================================================================
# CELL 9 / 10 : EXTERNAL VALIDATION — CheXpert official validation split
#   CheXpert has NO reports -> this is NOT a trimodal external validation.
#   We report: M7 with text zeroed, and the natively text-free M1 and M4.
#   Burden = 0 (no prior-study history available in the validation split).
# =============================================================================
cx = pd.read_csv(CHEXPERT_VALID)
cx['patient'] = cx['Path'].str.extract(r'(patient\d+)', expand=False)
cx['study'] = cx['Path'].str.extract(r'(patient\d+/study\d+)', expand=False)
print(f"CheXpert valid: {len(cx)} images | {cx['study'].nunique()} studies | {cx['patient'].nunique()} patients")
print('   <- use these exact counts (earlier text called 234 "studies"; check which they are)')


class CheXpertDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = tf_eval(Image.open(f"{CHEXPERT_ROOT}/{r['Path']}").convert('RGB'))
        except Exception:
            img = torch.zeros(3, 224, 224)
        fl, appa = str(r.get('Frontal/Lateral', '')).upper(), str(r.get('AP/PA', '')).upper()
        clin = torch.tensor([1.0 if appa == 'PA' else 0.0, 1.0 if appa == 'AP' else 0.0,
                             1.0 if fl == 'LATERAL' else 0.0, 0.0], dtype=torch.float32)
        y = torch.tensor([0.0 if pd.isna(r.get(d, 0)) or r.get(d, 0) == -1 else float(r.get(d, 0))
                          for d in DISEASES], dtype=torch.float32)
        z = torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
        return img, z, z, clin, y


cx_loader = DataLoader(CheXpertDataset(cx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def cx_aggregate(Y, L):
    sid = cx['study'].values
    Lg = pd.DataFrame(L).groupby(sid, sort=True).mean()
    Yg = pd.DataFrame(Y).groupby(sid, sort=True).max()
    pat = pd.Series(cx['patient'].values, index=sid).groupby(level=0).first().loc[Lg.index].values
    return Yg.values, sigmoid(Lg.values), pat


ext_models = [('M7_AGCMF (text zeroed)', 'M7_AGCMF', dict(zero_text=True)),
              ('M4_Img+Clin', 'M4_Img+Clin', {}), ('M1_Image', 'M1_Image', {})]
ext_rows, ext_class = [], None
for label, name, flags in ext_models:
    if not os.path.exists(f'{OUT_DIR}/{name}_best.pt'):
        print(f'skip {label} (not trained)'); continue
    mdl = FusionNet(**MODEL_ZOO[name]).to(DEVICE)
    mdl.load_state_dict(torch.load(f'{OUT_DIR}/{name}_best.pt', map_location=DEVICE))
    Y, L = predict_logits(mdl, cx_loader, **flags)
    Yg, Pg, pat = cx_aggregate(Y, L)
    pos = Yg.sum(0).astype(int)
    aucs = per_class_auc(Yg, Pg)
    if ext_class is None:
        ext_class = pd.DataFrame({'disease': DISEASES, 'positives': pos})
    ext_class[label] = np.round(aucs, 3)
    adequate = pos >= 10
    m_all = float(np.nanmean(aucs)); m_ad = float(np.nanmean(np.where(adequate, aucs, np.nan)))
    b = cluster_bootstrap(lambda idx: float(np.nanmean(np.where(adequate, per_class_auc(Yg[idx], Pg[idx]), np.nan))),
                          pat, n_boot=1000)
    lo, hi = ci(b)
    ext_rows.append({'model': label, 'n_studies': len(Yg),
                     'mean_AUC_classes_ge10_pos': round(m_ad, 3), '95CI': f'{lo:.3f}–{hi:.3f}',
                     'n_classes_ge10_pos': int(adequate.sum()),
                     'mean_AUC_all_valid_classes': round(m_all, 3)})
    del mdl; torch.cuda.empty_cache()

ext_df = pd.DataFrame(ext_rows)
ext_df.to_csv(f'{OUT_DIR}/external_chexpert_summary.csv', index=False)
ext_class.to_csv(f'{OUT_DIR}/external_chexpert_per_class.csv', index=False)
print('\n=== External CheXpert (image + view features only; NOT trimodal) ===')
print(ext_df.to_string(index=False))
print(ext_class.to_string(index=False))
print('\nCELL 9 done.')


# =============================================================================
# CELL 10 / 10 : ONE SUMMARY FILE with every number the manuscript needs
#   -> send me  rev5_results/REVISION_NUMBERS.md  and  rev5_results.zip
# =============================================================================
import shutil

L_ = []
w = L_.append
w('# AGCMF — corrected-subset results (auto-generated, do not edit numbers by hand)\n')
w('## Data'); w(split_summary.to_string(index=False))
w(f"\nEvaluation unit: {'study (image logits averaged)' if AGGREGATE_TO_STUDY else 'image'}; "
  f"test n = {len(d7['tY'])}, patients = {len(np.unique(d7['tpat']))}")
w(f"\nBurden score: {json.dumps(burden_stats)}")

w('\n## Headline (Abstract) — M7-AGCMF, test set, 95% patient-level bootstrap CI')
r7 = table6[table6.Model == 'M7_AGCMF'].iloc[0]
for k in ['mean_AUC', 'macro_F1', 'sensitivity', 'specificity', 'accuracy']:
    w(f'- {k}: {r7[k]:.3f} (95% CI {r7[k + "_CI"]})')

w('\n## Table 6 (corrected ablation)'); w(table6.to_string(index=False))
w('\n## Table 7 (per-disease AUC)'); w(t7.to_string(index=False))
w('\n## Table 8 (significance)'); w(table8.to_string(index=False))
w('\n## Calibration (macro, test)'); w(pd.DataFrame(macro).T.to_string())
w('\n## Missing-modality (M7, inference-time)'); w(mm_df.to_string(index=False))
w('\n## MC-Dropout'); w(mc_df.to_string(index=False))
w(f'\nGate stats: {json.dumps(gate_stats)}')
w('\n## External CheXpert (image + view only)'); w(ext_df.to_string(index=False))
w(ext_class.to_string(index=False))
w('\n## Parameters'); w(params_df.to_string(index=False))
infos = [json.load(open(f'{OUT_DIR}/{n}_info.json')) for n in TABLE_ORDER if os.path.exists(f'{OUT_DIR}/{n}_info.json')]
w('\n## Training'); w(pd.DataFrame(infos).to_string(index=False))
w('\n## Environment'); w(json.dumps(env, indent=2))

with open(f'{OUT_DIR}/REVISION_NUMBERS.md', 'w') as f:
    f.write('\n'.join(L_))
print('\n'.join(L_))

# zip everything except model weights (for the paper + GitHub)
tmp = '/content/rev5_export'
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for fn in os.listdir(OUT_DIR):
    if not fn.endswith('.pt'):
        shutil.copy(f'{OUT_DIR}/{fn}', tmp)
shutil.make_archive(f'{BASE_PATH}/rev5_results', 'zip', tmp)
print(f'\nSaved: {OUT_DIR}/REVISION_NUMBERS.md  and  {BASE_PATH}/rev5_results.zip')
print('Send both files back so the manuscript + response letter can be rewritten with these numbers.')


