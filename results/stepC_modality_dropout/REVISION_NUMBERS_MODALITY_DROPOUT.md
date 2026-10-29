# Modality-dropout experiment (p_drop = 0.3)

```
=== Modality-dropout experiment: test mean AUC (95% CI), 735 studies ===
        scenario            M7_AGCMF           M6_BiLSTM         M7_AGCMF_MD        M6_BiLSTM_MD dedicated_model dedicated_AUC dAUC_M7MD_vs_M6MD_pp  p_raw  p_Holm
   Full trimodal 0.956 (0.943–0.967) 0.946 (0.924–0.965) 0.940 (0.923–0.957) 0.943 (0.924–0.958)        M7_AGCMF         0.956   -0.3 (-1.4 to 1.1)  0.728   1.000
    Image + Text 0.957 (0.944–0.967) 0.945 (0.923–0.965) 0.940 (0.923–0.956) 0.943 (0.924–0.959)     M5_Img+Text         0.959   -0.3 (-1.4 to 1.1)  0.717   1.000
 Text + Clinical 0.956 (0.943–0.967) 0.945 (0.924–0.965) 0.939 (0.922–0.956) 0.940 (0.921–0.957)               —             —   -0.1 (-1.3 to 1.2)  0.914   1.000
Image + Clinical 0.673 (0.644–0.704) 0.650 (0.619–0.680) 0.713 (0.684–0.739) 0.726 (0.697–0.753)     M4_Img+Clin         0.741   -1.3 (-3.3 to 0.8)  0.217   1.000
       Text only 0.957 (0.944–0.967) 0.943 (0.920–0.964) 0.939 (0.922–0.956) 0.940 (0.921–0.957)         M2_Text         0.956   -0.1 (-1.2 to 1.2)  0.954   1.000
      Image only 0.664 (0.636–0.693) 0.640 (0.609–0.669) 0.703 (0.674–0.731) 0.728 (0.699–0.756)        M1_Image         0.737  -2.6 (-4.8 to -0.3)  0.028   0.196
   Clinical only 0.588 (0.563–0.618) 0.611 (0.584–0.639) 0.647 (0.616–0.675) 0.646 (0.615–0.675)     M3_Clinical         0.646    0.1 (-2.1 to 2.6)  0.904   1.000

=== AGCMF gate by input condition (does it adapt?) ===
      model         scenario  gate_mean  gate_sd
M7_AGCMF_MD    Full trimodal      0.510    0.062
M7_AGCMF_MD Image + Clinical      0.570    0.050
M7_AGCMF_MD       Image only      0.566    0.051

=== Full-input metrics (MD models) ===
       model  mean_AUC  macro_F1  sensitivity  specificity  accuracy   PPV   NPV
 M7_AGCMF_MD     0.940     0.613        0.851        0.902     0.899 0.545 0.981
M6_BiLSTM_MD     0.943     0.630        0.821        0.925     0.916 0.581 0.979

=== External CheXpert (no reports) incl. modality-dropout models ===
                     model  n_studies  mean_AUC_classes_ge10_pos        95CI
 M7_AGCMF_MD (text zeroed)        200                      0.776 0.740–0.809
M6_BiLSTM_MD (text zeroed)        200                      0.793 0.761–0.820
    M7_AGCMF (text zeroed)        200                      0.702 0.664–0.736
               M4_Img+Clin        200                      0.805 0.778–0.829
                  M1_Image        200                      0.759 0.726–0.790

```
