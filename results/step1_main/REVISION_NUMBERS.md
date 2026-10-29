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
- mean_AUC: 0.956 (95% CI 0.943–0.968)
- macro_F1: 0.661 (95% CI 0.633–0.684)
- sensitivity: 0.870 (95% CI 0.831–0.901)
- specificity: 0.934 (95% CI 0.923–0.945)
- accuracy: 0.929 (95% CI 0.918–0.940)

## Table 6 (corrected ablation)
      Model                         Modality  mean_AUC mean_AUC_CI  macro_F1 macro_F1_CI  sensitivity sensitivity_CI  specificity specificity_CI  accuracy accuracy_CI  dAUC_vs_M1_pp
   M1_Image                       Image only     0.737 0.708–0.764     0.323 0.300–0.344        0.588    0.536–0.633        0.730    0.701–0.762     0.725 0.701–0.753            0.0
    M2_Text                        Text only     0.956 0.939–0.970     0.685 0.655–0.705        0.868    0.829–0.902        0.939    0.930–0.949     0.933 0.924–0.942           21.9
M3_Clinical                    Clinical only     0.646 0.614–0.676     0.259 0.231–0.283        0.575    0.511–0.630        0.608    0.570–0.649     0.603 0.576–0.636           -9.1
M4_Img+Clin                 Image + Clinical     0.741 0.712–0.768     0.333 0.311–0.351        0.636    0.580–0.682        0.715    0.677–0.757     0.711 0.680–0.747            0.4
M5_Img+Text                     Image + Text     0.959 0.946–0.971     0.688 0.660–0.709        0.854    0.819–0.889        0.942    0.932–0.952     0.935 0.926–0.945           22.2
  M6_BiLSTM Image + Text + Clinical (BiLSTM)     0.946 0.924–0.964     0.665 0.639–0.686        0.840    0.791–0.883        0.943    0.934–0.953     0.934 0.926–0.944           20.8
   M7_AGCMF  Image + Text + Clinical (AGCMF)     0.956 0.943–0.968     0.661 0.633–0.684        0.870    0.831–0.901        0.934    0.923–0.945     0.929 0.918–0.940           21.9

## Table 7 (per-disease AUC)
                   Disease  test_positives  M1_Image  M2_Text  M3_Clinical  M4_Img+Clin  M5_Img+Text  M6_BiLSTM  M7_AGCMF
                No Finding             221     0.849    0.979        0.765        0.854        0.980      0.983     0.975
Enlarged Cardiomediastinum              35     0.698    0.945        0.649        0.697        0.917      0.914     0.890
              Cardiomegaly             142     0.772    0.962        0.681        0.771        0.961      0.953     0.953
              Lung Opacity             163     0.733    0.969        0.660        0.737        0.970      0.964     0.970
               Lung Lesion              19     0.485    0.972        0.400        0.566        0.927      0.932     0.944
                     Edema              79     0.863    0.984        0.754        0.880        0.989      0.990     0.987
             Consolidation              35     0.823    0.935        0.666        0.789        0.958      0.954     0.964
                 Pneumonia              45     0.670    0.950        0.565        0.665        0.955      0.956     0.943
               Atelectasis             151     0.748    0.976        0.665        0.744        0.974      0.968     0.971
              Pneumothorax              49     0.728    0.972        0.661        0.762        0.972      0.975     0.972
          Pleural Effusion             169     0.872    0.980        0.723        0.874        0.977      0.975     0.975
             Pleural Other              12     0.717    0.905        0.514        0.704        0.969      0.840     0.938
                  Fracture              13     0.546    0.889        0.562        0.503        0.912      0.876     0.930
           Support Devices             216     0.818    0.964        0.784        0.826        0.964      0.957     0.972

## Table 8 (significance)
       Comparison  AUC_M7  AUC_base  dAUC_pp dAUC_95CI_pp  p_bootstrap DeLong_sig_classes_Holm  Wilcoxon_W  Wilcoxon_p  p_bootstrap_Holm  significant_after_Holm
   M7 vs M1_Image   0.956     0.737     21.9 19.2 to 24.9       0.0005                   14/14         0.0    0.000122             0.003                    True
    M7 vs M2_Text   0.956     0.956     -0.0  -1.4 to 1.5       0.9600                    0/14        50.0    0.903198             1.000                   False
M7 vs M3_Clinical   0.956     0.646     31.0 27.8 to 34.4       0.0005                   14/14         0.0    0.000122             0.003                    True
M7 vs M4_Img+Clin   0.956     0.741     21.5 18.5 to 24.6       0.0005                   14/14         0.0    0.000122             0.003                    True
M7 vs M5_Img+Text   0.956     0.959     -0.3  -1.3 to 0.6       0.5500                    0/14        42.0    0.541626             1.000                   False
  M7 vs M6_BiLSTM   0.956     0.946      1.0  -0.7 to 2.9       0.2720                    0/14        38.0    0.390991             0.816                   False

## Calibration (macro, test)
          brier       ece     slope intercept     brier_95CI
raw    0.037157  0.032853  0.502021 -0.839657  0.0319–0.0421
temp   0.036156  0.028285  0.860978 -0.839051            NaN
platt  0.032379  0.016065  0.950151  0.031115  0.0275–0.0367

## Missing-modality (M7, inference-time)
        scenario  mean_AUC    AUC_95CI  macro_F1
   Full trimodal     0.956 0.943–0.967     0.661
    Image + Text     0.957 0.944–0.967     0.649
 Text + Clinical     0.956 0.943–0.967     0.644
Image + Clinical     0.673 0.644–0.704     0.195
       Text only     0.957 0.944–0.967     0.612
      Image only     0.664 0.636–0.693     0.218
   Clinical only     0.588 0.563–0.618     0.212

## MC-Dropout
                   disease  mean_predictive_SD  AUROC_SD_detects_error
                No Finding              0.0224                   0.851
Enlarged Cardiomediastinum              0.0300                   0.962
              Cardiomegaly              0.0546                   0.844
              Lung Opacity              0.0324                   0.836
               Lung Lesion              0.0154                   0.964
                     Edema              0.0199                   0.938
             Consolidation              0.0092                   0.936
                 Pneumonia              0.0311                   0.961
               Atelectasis              0.0381                   0.826
              Pneumothorax              0.0179                   0.899
          Pleural Effusion              0.0358                   0.867
             Pleural Other              0.0060                   0.981
                  Fracture              0.0073                   0.971
           Support Devices              0.0392                   0.847

Gate stats: {"gate_mean": 0.46686944365501404, "gate_sd": 0.05907045677304268, "gate_p5": 0.39281854033470154, "gate_p95": 0.5600674152374268}

## External CheXpert (image + view only)
                 model  n_studies  mean_AUC_classes_ge10_pos        95CI  n_classes_ge10_pos  mean_AUC_all_valid_classes
M7_AGCMF (text zeroed)        200                      0.702 0.664–0.736                   9                       0.687
           M4_Img+Clin        200                      0.805 0.778–0.829                   9                       0.764
              M1_Image        200                      0.759 0.726–0.790                   9                       0.722
                   disease  positives  M7_AGCMF (text zeroed)  M4_Img+Clin  M1_Image
                No Finding         26                   0.588        0.894     0.813
Enlarged Cardiomediastinum        105                   0.733        0.749     0.657
              Cardiomegaly         66                   0.698        0.761     0.752
              Lung Opacity        116                   0.652        0.850     0.813
               Lung Lesion          1                   0.085        0.251     0.080
                     Edema         42                   0.734        0.861     0.852
             Consolidation         32                   0.734        0.845     0.663
                 Pneumonia          8                   0.810        0.888     0.788
               Atelectasis         75                   0.719        0.779     0.792
              Pneumothorax          6                   0.763        0.740     0.697
          Pleural Effusion         64                   0.757        0.773     0.770
             Pleural Other          1                   0.955        0.809     0.990
                  Fracture          0                     NaN          NaN       NaN
           Support Devices         99                   0.702        0.733     0.716

## Parameters
      model  params_M
   M1_Image     24.07
    M2_Text    108.54
M3_Clinical      0.04
M4_Img+Clin     24.20
M5_Img+Text    132.71
  M6_BiLSTM    132.97
   M7_AGCMF    133.30

## Training
      model  best_epoch  best_val_mean_auc  train_minutes_total  params_M
   M1_Image          10           0.744096                  7.6     24.07
    M2_Text           4           0.967979                  9.2    108.54
M3_Clinical           7           0.661844                  1.4      0.04
M4_Img+Clin           6           0.739182                  7.8     24.20
M5_Img+Text           4           0.968293                 15.3    132.71
  M6_BiLSTM          10           0.963402                 15.7    132.97
   M7_AGCMF           8           0.962628                 15.5    133.30

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
  "evaluation_unit": "study"
}

