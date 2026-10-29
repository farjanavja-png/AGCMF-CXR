# =============================================================================
# 04_representativeness_check.py
# Compares the corrected image-available subset with the full MIMIC-CXR cohort
# (disease prevalence at study level, view position at image level) using
# standardised mean differences (SMD). Produces Supplementary Table S2.
# No GPU, no training. Run in Colab:  %run 04_representativeness_check.py
# =============================================================================
import os
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

BASE = '/content/drive/MyDrive/test'          # folder with the MIMIC-CXR csv files
OUT = f'{BASE}/rev5_results'
DISEASES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

full = pd.read_csv(f'{BASE}/mimic-cxr-2.0.0-chexpert.csv')
sub_img = pd.read_csv(f'{BASE}/available_studies_filtered.csv')
sub = sub_img.drop_duplicates('study_id')
meta = pd.read_csv(f'{BASE}/metadata.csv')

print(f"Full cohort : {full.study_id.nunique():,} studies, {full.subject_id.nunique():,} patients")
print(f"Subset      : {sub.study_id.nunique():,} studies, {len(sub_img):,} images, "
      f"{sub.subject_id.nunique():,} patients\n")


def smd(p1, p2):
    d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p2 - p1) / d if d > 0 else np.nan


rep = pd.DataFrame([{'Disease': d,
                     'Full cohort (%)': round(100 * (full[d] == 1).mean(), 1),
                     'Subset (%)': round(100 * (sub[d] == 1).mean(), 1),
                     'SMD': round(smd((full[d] == 1).mean(), (sub[d] == 1).mean()), 3)}
                    for d in DISEASES])

vf = meta['ViewPosition'].fillna('Missing').value_counts(normalize=True)
vs = sub_img['ViewPosition'].fillna('Missing').value_counts(normalize=True)
views = pd.DataFrame({'Full cohort (%)': (100 * vf).round(1), 'Subset (%)': (100 * vs).round(1)}).fillna(0)
views['SMD'] = [round(smd(vf.get(v, 0), vs.get(v, 0)), 3) for v in views.index]
views = views.sort_values('Full cohort (%)', ascending=False).head(6)

print('=== Supplementary Table S2a: disease prevalence (study level) ===')
print(rep.to_string(index=False))
print(f"\nMax |SMD| = {rep.SMD.abs().max():.3f}  (|SMD| < 0.1 = negligible)\n")
print('=== Supplementary Table S2b: view position (image level) ===')
print(views.to_string())

os.makedirs(OUT, exist_ok=True)
rep.to_csv(f'{OUT}/supp_table_S2a_prevalence.csv', index=False)
views.to_csv(f'{OUT}/supp_table_S2b_views.csv')
