# AGCMF — Adaptive Gated Cross-Modal Fusion for Trimodal Multi-Label Chest X-ray Classification

Code and result summaries for the manuscript
**"Adaptable Gated Cross-Modal Fusion for Trimodal Multi-Label Chest X-ray Classification"**
(Shaik Farjana, C S Pavan Kumar).

This repository contains the complete, leakage-controlled experimental pipeline used in the
revised manuscript: data splitting, leak-free clinical features, training of seven ablation
models (M1–M7), statistical comparison, calibration, uncertainty estimation, missing-modality
analysis and external validation on CheXpert.

> **No patient data are included.** MIMIC-CXR and CheXpert are credentialed datasets and must be
> obtained directly from their providers (see *Data access*).

---

## Repository structure

```
scripts/
  01_step1_main_experiments.py      # main corrected experiment (M1–M7, Tables 6–9, calibration,
                                    #   missing-modality, MC-Dropout, CheXpert external validation)
  02_stepC_modality_dropout.py      # modality-dropout training of M6 and M7 (robustness to missing inputs)
  03_stepD_preread_text_xrv.py      # clinically realistic task: pre-interpretation text only
                                    #   (indication/history + prior report) and a chest-X-ray-pretrained encoder
  04_representativeness_check.py    # subset vs full cohort comparison (Supplementary Table S2)
splits/
  split_assignment.csv              # patient-level train/val/test assignment (subject_id, study_id, dicom_id)
  split_summary.csv                 # images / studies / patients per split
results/
  step1_main/REVISION_NUMBERS.md                     # all numbers of the main analysis
  stepC_modality_dropout/REVISION_NUMBERS_MODALITY_DROPOUT.md
  stepD_preread_xrv/REVISION_NUMBERS.md
  supplementary_table_S2_representativeness.md
requirements.txt
```

## Models

| ID | Inputs | Fusion |
|---|---|---|
| M1 | Image | — |
| M2 | Report text | — |
| M3 | Structured clinical features | — |
| M4 | Image + clinical | Concatenation |
| M5 | Image + text | Concatenation |
| M6 | Image + text + clinical | BiLSTM over modality tokens |
| M7 | Image + text + clinical | **AGCMF** (cross-modal attention + self-attention + per-sample gate) |

All models share the same encoders (ResNet-50 / Bio_ClinicalBERT / clinical MLP in Steps 1 and C;
TorchXRayVision DenseNet-121 (NIH weights) in Step D), classification head, loss, optimiser, seed,
epochs and augmentation. Checkpoints are selected on validation only.

## Data and leakage controls

* **Data:** all locally available MIMIC-CXR-JPG v2.0.0 studies — 5,227 studies / 8,502 images /
  1,500 patients. The subset matches the full cohort (227,827 studies) closely: maximum |SMD|
  0.028 for disease prevalence and 0.031 for view position (Supplementary Table S2).
* **Split:** patient-level 70/15/15 (GroupShuffleSplit, seed 42); no patient appears in more than
  one split (asserted in code). Test set: 735 studies, 225 patients.
* **Structured features (4):** PA, AP, lateral view flags and a history-based disease-burden score.
  The burden score uses only the patient's studies with a **strictly earlier** timestamp
  (study-level, ties excluded); it is verified against a brute-force recomputation in code.
  Earlier label-duplicating flags (No Finding, Support Devices) were removed.
* **Labels:** CheXpert labels (uncertain/blank → negative).
* **Evaluation unit:** study (image logits averaged within a study).
* **Statistics:** patient-level cluster bootstrap 95% CIs (2,000 replicates); paired bootstrap of
  mean-AUC differences with Holm correction; per-class DeLong tests with Holm correction.
* **Text:** Step 1 uses FINDINGS + IMPRESSION of the current report. Because the CheXpert labels
  are extracted from this same text, these results represent an upper bound. Step D uses only
  text available before image interpretation (INDICATION/HISTORY of the current report and the
  IMPRESSION of the patient's previous report).

## Main results (test set, 735 studies, mean AUC with 95% CI)

| Model | Step 1 (current report text) | Step D (pre-read text, X-ray encoder) |
|---|---|---|
| M1 Image | 0.737 (0.708–0.764) | 0.726 (0.693–0.755) |
| M2 Text | 0.956 (0.939–0.970) | 0.727 (0.695–0.755) |
| M3 Clinical | 0.646 (0.614–0.676) | 0.646 (0.614–0.676) |
| M4 Image + clinical | 0.741 (0.712–0.768) | 0.729 (0.700–0.756) |
| M5 Image + text | 0.959 (0.946–0.971) | 0.768 (0.734–0.797) |
| M6 BiLSTM | 0.946 (0.924–0.964) | 0.725 (0.691–0.752) |
| M7 AGCMF | 0.956 (0.943–0.968) | 0.741 (0.707–0.769) |

In both settings AGCMF was significantly better than unimodal clinical input, and not
significantly different from simple image+text fusion or BiLSTM fusion after Holm correction.
Full tables, per-disease AUCs, significance tests, calibration (raw and Platt-scaled),
MC-Dropout uncertainty, missing-modality and external CheXpert results are in `results/`.

## Data access

* **MIMIC-CXR-JPG v2.0.0** — https://physionet.org/content/mimic-cxr-jpg/ (credentialed access,
  CITI training and data use agreement required). Free-text reports: https://physionet.org/content/mimic-cxr/
* **CheXpert (v1.0-small, official validation split)** — https://stanfordmlgroup.github.io/competitions/chexpert/

Expected Google Drive layout (edit `BASE_PATH` in `01_step1_main_experiments.py` if different):

```
MyDrive/test/
  metadata.csv                         # MIMIC-CXR-JPG metadata (must include StudyDate, StudyTime, ViewPosition)
  mimic-cxr-2.0.0-chexpert.csv         # CheXpert labels for MIMIC-CXR
  available_studies_filtered.csv       # rows (images) whose JPG is available locally (created by the image scan)
  p10 (1)/pXXXXXXXX/sYYYYYYYY/*.jpg    # images
  p10/pXXXXXXXX/sYYYYYYYY.txt          # reports
MyDrive/CheXpert-v1.0-small/valid.csv  # + images
```

## How to run (Google Colab, T4 GPU)

```python
%run /content/drive/MyDrive/test/01_step1_main_experiments.py      # ~1.5–2 h
%run /content/drive/MyDrive/test/02_stepC_modality_dropout.py      # ~50 min (needs script 01 in the same folder)
%run /content/drive/MyDrive/test/03_stepD_preread_text_xrv.py      # ~1.5–2 h (needs script 01 in the same folder)
%run /content/drive/MyDrive/test/04_representativeness_check.py    # ~2 min
```

> Scripts 02 and 03 locate script 01 by its original file name `AGCMF_Rev5_Step1.py`. When
> running from this repository, either copy `01_step1_main_experiments.py` to
> `MyDrive/test/AGCMF_Rev5_Step1.py`, or edit the `MAIN_SCRIPT` / `STEP1` path at the top of
> scripts 02 and 03.

All scripts are resumable: if the Colab session disconnects, re-run the same command — finished
models are reloaded and unfinished models resume from the last completed epoch.

## Reproducibility

Python 3.13, PyTorch 2.11 (CUDA 13.0), torchvision 0.26, transformers 5.20, torchxrayvision 1.5.7,
single NVIDIA Tesla T4 (15.6 GB). Seed 42; batch size 16; 10 epochs; AdamW (lr 1e-4 for
fusion/classification layers, 2e-5 for Bio_ClinicalBERT), weight decay 1e-4, gradient clipping 1.0,
fp16 mixed precision; best checkpoint by validation mean AUC. Augmentation: horizontal flip (p=0.5),
rotation ±10°, brightness/contrast jitter ±0.2, translation ±5%. Training time per model: 1–20 min.

## Citation

If you use this code, please cite the article (citation details will be added on publication).

## License

Code: MIT License (see `LICENSE`). The datasets are subject to their own licences and data use agreements.
