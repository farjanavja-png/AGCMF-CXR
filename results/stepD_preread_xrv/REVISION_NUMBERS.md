# AGCMF — corrected-subset results (auto-generated, do not edit numbers by hand)

## Data
split  images  studies  patients
train    5752     3525      1049
  val    1542      967       226
 test    1208      735       225
total    8502     5227      1500

Evaluation unit: study (image logits averaged); test n = 735, patients = 225

Burden score: {"studies_with_timestamps": 52431, "studies_with_prior_history_pct_all": 71.4, "subset_studies": 5227, "subset_nonzero_burden_pct": 61.4, "subset_burden_mean": 0.0916, "corr_burden_vs_CURRENT_pos_count": 0.512}

## Headline (Abstract) — M7-AGCMF, test set, 95% patient-level bootstrap CI
- mean_AUC: 0.741 (95% CI 0.707–0.769)
- macro_F1: 0.339 (95% CI 0.307–0.366)
- sensitivity: 0.679 (95% CI 0.622–0.721)
- specificity: 0.686 (95% CI 0.653–0.724)
- accuracy: 0.698 (95% CI 0.671–0.731)

## Table 6 (corrected ablation)
      Model                         Modality  mean_AUC mean_AUC_CI  macro_F1 macro_F1_CI  sensitivity sensitivity_CI  specificity specificity_CI  accuracy accuracy_CI  dAUC_vs_M1_pp
   M1_Image                       Image only     0.726 0.693–0.755     0.317 0.295–0.337        0.679    0.627–0.724        0.655    0.614–0.700     0.661 0.628–0.700            0.0
    M2_Text                        Text only     0.727 0.695–0.755     0.346 0.313–0.372        0.605    0.544–0.651        0.745    0.717–0.775     0.748 0.726–0.773            0.2
M3_Clinical                    Clinical only     0.646 0.614–0.676     0.259 0.231–0.283        0.575    0.511–0.630        0.608    0.570–0.649     0.603 0.576–0.636           -7.9
M4_Img+Clin                 Image + Clinical     0.729 0.700–0.756     0.315 0.292–0.335        0.649    0.586–0.703        0.664    0.623–0.709     0.669 0.637–0.706            0.3
M5_Img+Text                     Image + Text     0.768 0.734–0.797     0.350 0.323–0.373        0.649    0.593–0.693        0.718    0.687–0.753     0.718 0.693–0.748            4.2
  M6_BiLSTM Image + Text + Clinical (BiLSTM)     0.725 0.691–0.752     0.325 0.299–0.345        0.627    0.558–0.675        0.693    0.654–0.737     0.696 0.662–0.733           -0.0
   M7_AGCMF  Image + Text + Clinical (AGCMF)     0.741 0.707–0.769     0.339 0.307–0.366        0.679    0.622–0.721        0.686    0.653–0.724     0.698 0.671–0.731            1.5

## Table 7 (per-disease AUC)
                   Disease  test_positives  M1_Image  M2_Text  M3_Clinical  M4_Img+Clin  M5_Img+Text  M6_BiLSTM  M7_AGCMF
                No Finding             221     0.848    0.777        0.765        0.848        0.825      0.808     0.813
Enlarged Cardiomediastinum              35     0.657    0.696        0.649        0.682        0.701      0.679     0.667
              Cardiomegaly             142     0.783    0.696        0.681        0.780        0.721      0.698     0.718
              Lung Opacity             163     0.732    0.677        0.660        0.716        0.713      0.689     0.714
               Lung Lesion              19     0.585    0.709        0.400        0.592        0.755      0.652     0.695
                     Edema              79     0.857    0.788        0.754        0.857        0.833      0.798     0.817
             Consolidation              35     0.788    0.732        0.666        0.761        0.781      0.732     0.728
                 Pneumonia              45     0.643    0.681        0.565        0.631        0.739      0.640     0.690
               Atelectasis             151     0.761    0.673        0.665        0.743        0.731      0.702     0.705
              Pneumothorax              49     0.710    0.863        0.661        0.754        0.875      0.854     0.848
          Pleural Effusion             169     0.857    0.764        0.723        0.841        0.825      0.823     0.803
             Pleural Other              12     0.670    0.640        0.514        0.704        0.629      0.640     0.578
                  Fracture              13     0.491    0.636        0.562        0.502        0.772      0.582     0.754
           Support Devices             216     0.776    0.852        0.784        0.795        0.850      0.855     0.843

## Table 8 (significance)
       Comparison  AUC_M7  AUC_base  dAUC_pp dAUC_95CI_pp  p_bootstrap DeLong_sig_classes_Holm  Wilcoxon_W  Wilcoxon_p  p_bootstrap_Holm  significant_after_Holm
   M7 vs M1_Image   0.741     0.726      1.5  -1.2 to 4.4       0.3230                    6/14        50.0    0.903198             0.678                   False
    M7 vs M2_Text   0.741     0.727      1.3  -0.7 to 3.5       0.2260                    1/14        32.0    0.216553             0.678                   False
M7 vs M3_Clinical   0.741     0.646      9.4  6.4 to 12.3       0.0005                    5/14         0.0    0.000122             0.003                    True
M7 vs M4_Img+Clin   0.741     0.729      1.2  -1.4 to 3.7       0.3940                    3/14        51.0    0.951538             0.678                   False
M7 vs M5_Img+Text   0.741     0.768     -2.7 -5.0 to -0.4       0.0260                    0/14         1.0    0.000244             0.130                   False
  M7 vs M6_BiLSTM   0.741     0.725      1.6  -0.4 to 3.6       0.0990                    0/14        39.0    0.426270             0.396                   False

## Calibration (macro, test)
          brier       ece     slope intercept     brier_95CI
raw    0.150656  0.188672  0.642523 -1.938224  0.1371–0.1620
temp   0.142763  0.177644  0.948856 -1.938886            NaN
platt  0.087438  0.024031  0.968219   0.06219  0.0789–0.0950

## Missing-modality (M7, inference-time)
        scenario  mean_AUC    AUC_95CI  macro_F1
   Full trimodal     0.741 0.706–0.768     0.339
    Image + Text     0.744 0.711–0.772     0.330
 Text + Clinical     0.716 0.682–0.744     0.316
Image + Clinical     0.700 0.666–0.729     0.252
       Text only     0.708 0.674–0.735     0.296
      Image only     0.697 0.664–0.725     0.220
   Clinical only     0.612 0.583–0.644     0.089

## MC-Dropout
                   disease  mean_predictive_SD  AUROC_SD_detects_error
                No Finding              0.0564                   0.753
Enlarged Cardiomediastinum              0.0569                   0.854
              Cardiomegaly              0.0736                   0.651
              Lung Opacity              0.0681                   0.537
               Lung Lesion              0.0389                   0.692
                     Edema              0.0556                   0.586
             Consolidation              0.0528                   0.885
                 Pneumonia              0.0612                   0.664
               Atelectasis              0.0583                   0.579
              Pneumothorax              0.0432                   0.823
          Pleural Effusion              0.0572                   0.579
             Pleural Other              0.0271                   0.946
                  Fracture              0.0313                   0.880
           Support Devices              0.0540                   0.636

Gate stats: {"gate_mean": 0.5622016191482544, "gate_sd": 0.05502217635512352, "gate_p5": 0.48372480273246765, "gate_p95": 0.6416653394699097}

## External CheXpert (image + view only)
                 model  n_studies  mean_AUC_classes_ge10_pos        95CI  n_classes_ge10_pos  mean_AUC_all_valid_classes
M7_AGCMF (text zeroed)        200                      0.798 0.769–0.823                   9                       0.818
           M4_Img+Clin        200                      0.791 0.758–0.817                   9                       0.787
              M1_Image        200                      0.793 0.763–0.818                   9                       0.782
                   disease  positives  M7_AGCMF (text zeroed)  M4_Img+Clin  M1_Image
                No Finding         26                   0.930        0.864     0.888
Enlarged Cardiomediastinum        105                   0.815        0.796     0.778
              Cardiomegaly         66                   0.753        0.759     0.758
              Lung Opacity        116                   0.855        0.859     0.863
               Lung Lesion          1                   0.874        0.513     0.678
                     Edema         42                   0.767        0.807     0.831
             Consolidation         32                   0.848        0.790     0.742
                 Pneumonia          8                   0.827        0.842     0.712
               Atelectasis         75                   0.781        0.798     0.800
              Pneumothorax          6                   0.817        0.796     0.723
          Pleural Effusion         64                   0.795        0.785     0.803
             Pleural Other          1                   0.935        0.970     0.920
                  Fracture          0                     NaN          NaN       NaN
           Support Devices         99                   0.643        0.657     0.668

## Parameters
      model  params_M
   M1_Image      7.26
    M2_Text    108.54
M3_Clinical      0.04
M4_Img+Clin      7.40
M5_Img+Text    115.90
  M6_BiLSTM    116.17
   M7_AGCMF    116.50

## Training
      model  best_epoch  best_val_mean_auc  train_minutes_total  params_M
   M1_Image          10           0.721496                  9.6      7.26
    M2_Text           3           0.742205                  9.7    108.54
M3_Clinical           7           0.661844                  1.3      0.04
M4_Img+Clin           8           0.730099                  9.7      7.40
M5_Img+Text           3           0.756034                 19.0    115.90
  M6_BiLSTM           4           0.748512                 19.7    116.17
   M7_AGCMF           3           0.751226                 19.6    116.50

## Environment
{
  "python": "3.13.16",
  "os": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "torch": "2.11.0+cu130",
  "torchvision": "0.26.0+cu130",
  "transformers": "5.20.2",
  "cuda": "13.0",
  "cudnn": 92700,
  "gpu": "Tesla T4",
  "gpu_mem_GB": 15.6,
  "seed": 42,
  "batch_size": 16,
  "epochs": 10,
  "lr_head": 0.0001,
  "lr_bert": 2e-05,
  "weight_decay": 0.0001,
  "grad_clip": 1.0,
  "max_text_len": 128,
  "mixed_precision_fp16": true,
  "bootstrap_replicates": 2000,
  "mc_dropout_passes": 20,
  "evaluation_unit": "study",
  "task": "pre-read text (current INDICATION/HISTORY + prior report IMPRESSION)",
  "image_encoder": "TorchXRayVision DenseNet-121 (densenet121-res224-nih, NIH ChestX-ray14 only)",
  "torchxrayvision": "1.5.7"
}


## Pre-read text statistics
```
=== Pre-read text (REPORT IN METHODS) ===
{
  "studies": 5227,
  "with_indication_pct": 97.7,
  "with_prior_study_pct": 71.3,
  "with_prior_report_text_pct": 71.3,
  "mean_tokens": 64.6,
  "truncated_at_max_len_pct": 13.8
}
Example input: [CLS] _ _ _ f with new onset ascites / / eval for infection [SEP] no prior report available. [SEP]
```
