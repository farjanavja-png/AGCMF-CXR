# =============================================================================
# AGCMF_Rev5_StepC_ModalityDropout.py
#
# PRE-SPECIFIED follow-up experiment (motivated by Step-1 results: standard
# training made M7 text-dependent — test AUC 0.673 without text vs 0.741 for
# a dedicated image+clinical model; CheXpert 0.702 vs 0.805).
#
# Protocol (identical for both models, fixed BEFORE seeing any result):
#   - models: M7 AGCMF and M6 BiLSTM (the two trimodal fusion models)
#   - modality dropout: during training each modality of each sample is
#     dropped independently with p = 0.3 (at least one modality always kept)
#   - same seed, epochs (10), optimiser, LRs, augmentation, split as Step 1
#   - checkpoint selection on VALIDATION only, criterion = mean of
#     val AUC(full input) and val AUC(image + clinical, no report)
#   - test: all 7 input scenarios + external CheXpert; ALL results reported
#
# HOW TO RUN (T4 GPU):
#   %run /content/drive/MyDrive/test/AGCMF_Rev5_StepC_ModalityDropout.py
#   - in the SAME session as Step 1: starts immediately
#   - in a NEW session: it first re-runs Step-1 cells 1–5 automatically
#     (trained Step-1 models are reloaded, not retrained; ~10 min of data prep)
# Output: rev5_results/REVISION_NUMBERS_MODALITY_DROPOUT.md
# =============================================================================
import os
MAIN_SCRIPT = '/content/drive/MyDrive/test/AGCMF_Rev5_Step1.py'
if 'PRED' not in globals() or 'cluster_bootstrap' not in globals():
    print('Step-1 objects not in memory -> running Step-1 cells 1-5 (no retraining)...')
    _src = open(MAIN_SCRIPT, encoding='utf-8').read()
    _cut = _src.index('# CELL 6 / 10')
    _cut = _src.rfind('# ====', 0, _cut)
    exec(compile(_src[:_cut], MAIN_SCRIPT, 'exec'), globals())
if 'cx_loader' not in globals() or 'ext_df' not in globals():
    print('Re-running Step-1 Cell 9 (CheXpert loader + Step-1 external results)...')
    _src = open(MAIN_SCRIPT, encoding='utf-8').read()
    _a = _src.rfind('# ====', 0, _src.index('# CELL 9 / 10'))
    _b = _src.rfind('# ====', 0, _src.index('# CELL 10 / 10'))
    exec(compile(_src[_a:_b], MAIN_SCRIPT, 'exec'), globals())

P_DROP = 0.3
MD_MODELS = {'M7_AGCMF_MD': 'M7_AGCMF', 'M6_BiLSTM_MD': 'M6_BiLSTM'}
MOD_ORDER = ('img', 'txt', 'clin')


class MDFusionNet(FusionNet):
    """FusionNet + per-sample modality mask (B x 3: img, txt, clin)."""

    def forward(self, image, input_ids, attn_mask, clinical,
                zero_image=False, zero_text=False, zero_clinical=False,
                return_gate=False, mod_mask=None):
        B, dev = clinical.size(0), clinical.device
        if mod_mask is None:
            mod_mask = torch.ones(B, 3, device=dev)
        mod_mask = mod_mask.clone()
        if zero_image:
            mod_mask[:, 0] = 0
        if zero_text:
            mod_mask[:, 1] = 0
        if zero_clinical:
            mod_mask[:, 2] = 0
        h_i = self.img_proj(self.img_backbone(image).flatten(1)).float() if mod_mask[:, 0].any() \
            else torch.zeros(B, EMBED_DIM, device=dev)
        if mod_mask[:, 1].any():
            cls = self.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
            h_t = self.txt_proj(cls).float()
        else:
            h_t = torch.zeros(B, EMBED_DIM, device=dev)
        h_c = self.clin_proj(clinical).float()
        toks = [h_i * mod_mask[:, 0:1], h_t * mod_mask[:, 1:2], h_c * mod_mask[:, 2:3]]
        gate = None
        if self.fusion == 'agcmf':
            tokens = torch.stack(toks, dim=1)
            cross_out, _ = self.cross_attn(tokens, tokens, tokens)
            refined, _ = self.self_attn(cross_out, cross_out, cross_out)
            pooled_tokens, pooled_refined = tokens.mean(dim=1), refined.mean(dim=1)
            gate = self.gate(torch.cat([pooled_tokens, pooled_refined, toks[0]], dim=1))
            fused = gate * pooled_tokens + (1 - gate) * pooled_refined
        else:
            out, _ = self.lstm(torch.stack(toks, dim=1))
            fused = out.mean(dim=1)
        logits = self.classifier(fused)
        return (logits, gate) if return_gate else logits


def sample_mod_mask(B, p=P_DROP, device=DEVICE):
    m = (torch.rand(B, 3, device=device) >= p).float()
    empty = m.sum(1) == 0
    if empty.any():
        keep = torch.randint(0, 3, (int(empty.sum()),), device=device)
        m[empty.nonzero(as_tuple=True)[0], keep] = 1.0
    return m


def train_md(name):
    base = MD_MODELS[name]
    pred_path = f'{OUT_DIR}/{name}_best.pt'
    info_path = f'{OUT_DIR}/{name}_info.json'
    if os.path.exists(info_path):
        print(f'[{name}] already trained — skipping.'); return
    print(f'\n================ {name}  (modality dropout p={P_DROP}) ================')
    set_seed(SEED)
    model = MDFusionNet(**MODEL_ZOO[base]).to(DEVICE)
    gen = torch.Generator(); gen.manual_seed(SEED)
    tl = make_loader(train_df, True, True, generator=gen)
    vl = make_loader(val_df, False, True)
    pos_weight = compute_pos_weight(tl.dataset.lab.numpy()).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bert_p = [p for n, p in model.named_parameters() if n.startswith('bert.')]
    other_p = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
    optimizer = optim.AdamW([{'params': other_p, 'lr': LR_HEAD}, {'params': bert_p, 'lr': LR_BERT}],
                            weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()
    last_ckpt = f'{OUT_DIR}/{name}_last.pt'
    start_ep, best, best_ep, log = 0, -1.0, -1, []
    if os.path.exists(last_ckpt):
        st = torch.load(last_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st['model']); optimizer.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler'])
        start_ep, best, best_ep, log = st['epoch'] + 1, st['best'], st['best_ep'], st['log']
        print(f'Resuming {name} from epoch {start_ep + 1}')
    for ep in range(start_ep, EPOCHS):
        gen.manual_seed(SEED + ep); set_seed(SEED + ep)
        model.train(); t0 = time.time(); run_loss = 0.0
        for batch in tl:
            img, ids, mask, clin, y = to_dev(batch)
            mm = sample_mod_mask(img.size(0))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(img, ids, mask, clin, mod_mask=mm)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); run_loss += loss.item()
        Yv, Lv = predict_logits(model, vl)
        Yv2, Lv2 = predict_logits(model, vl, zero_text=True)
        a_full, a_notext = mean_auc(Yv, sigmoid(Lv)), mean_auc(Yv2, sigmoid(Lv2))
        crit = (a_full + a_notext) / 2
        sec = time.time() - t0
        log.append({'epoch': ep + 1, 'train_loss': run_loss / len(tl), 'val_auc_full': a_full,
                    'val_auc_no_text': a_notext, 'criterion': crit, 'seconds': sec})
        flag = ''
        if crit > best:
            best, best_ep, flag = crit, ep + 1, '  <- best'
            torch.save(model.state_dict(), pred_path)
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': ep, 'best': best, 'best_ep': best_ep, 'log': log}, last_ckpt)
        print(f'[{name}] epoch {ep + 1}/{EPOCHS} loss={run_loss / len(tl):.4f} '
              f'val_full={a_full:.4f} val_no_text={a_notext:.4f} ({sec / 60:.1f} min){flag}')
    pd.DataFrame(log).to_csv(f'{OUT_DIR}/{name}_train_log.csv', index=False)
    info = {'model': name, 'p_drop': P_DROP, 'best_epoch': best_ep, 'best_val_criterion': best,
            'train_minutes_total': round(sum(r['seconds'] for r in log) / 60, 1)}
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'[{name}] DONE {info}')
    del model, optimizer; torch.cuda.empty_cache()


for _n in MD_MODELS:
    train_md(_n)

# ----------------------------- evaluation ------------------------------------
SCEN = {
    'Full trimodal':    dict(zero_image=False, zero_text=False, zero_clinical=False),
    'Image + Text':     dict(zero_image=False, zero_text=False, zero_clinical=True),
    'Text + Clinical':  dict(zero_image=True,  zero_text=False, zero_clinical=False),
    'Image + Clinical': dict(zero_image=False, zero_text=True,  zero_clinical=False),
    'Text only':        dict(zero_image=True,  zero_text=False, zero_clinical=True),
    'Image only':       dict(zero_image=False, zero_text=True,  zero_clinical=True),
    'Clinical only':    dict(zero_image=True,  zero_text=True,  zero_clinical=False),
}
# dedicated Step-1 model that natively matches each scenario (reference)
DEDICATED = {'Full trimodal': 'M7_AGCMF', 'Image + Text': 'M5_Img+Text', 'Text + Clinical': None,
             'Image + Clinical': 'M4_Img+Clin', 'Text only': 'M2_Text', 'Image only': 'M1_Image',
             'Clinical only': 'M3_Clinical'}

test_loader_img = make_loader(test_df, False, need_image=True)
val_loader_img = make_loader(val_df, False, need_image=True)


def load_md(name):
    m = MDFusionNet(**MODEL_ZOO[MD_MODELS[name]]).to(DEVICE)
    m.load_state_dict(torch.load(f'{OUT_DIR}/{name}_best.pt', map_location=DEVICE))
    m.eval()
    return m


SC_P = {}          # (model, scenario) -> (Y, P, patients)
gate_rows = []
for name in MD_MODELS:
    m = load_md(name)
    Yv, Lv = predict_logits(m, val_loader_img)
    vY, vL, _ = aggregate(Yv, Lv, val_df)
    thr = youden_thresholds(vY, sigmoid(vL))      # thresholds from full-input validation
    THR[name] = thr
    for sc, fl in SCEN.items():
        Y, L = predict_logits(m, test_loader_img, **fl)
        Yg, Lg, pat = aggregate(Y, L, test_df)
        SC_P[(name, sc)] = (Yg, sigmoid(Lg), pat)
    if MD_MODELS[name] == 'M7_AGCMF':
        for sc in ['Full trimodal', 'Image + Clinical', 'Image only']:
            gs = []
            with torch.no_grad():
                for batch in test_loader_img:
                    img, ids, mask, clin, y = to_dev(batch)
                    with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                        _, g = m(img, ids, mask, clin, return_gate=True, **SCEN[sc])
                    gs.append(g.float().mean(1).cpu().numpy())
            gs = np.concatenate(gs)
            gate_rows.append({'model': name, 'scenario': sc, 'gate_mean': round(float(gs.mean()), 3),
                              'gate_sd': round(float(gs.std()), 3)})
    del m; torch.cuda.empty_cache()

# also the Step-1 (no-dropout) M7 and M6 under the same scenarios, for the comparison table
for base in ['M7_AGCMF', 'M6_BiLSTM']:
    m = FusionNet(**MODEL_ZOO[base]).to(DEVICE)
    m.load_state_dict(torch.load(f'{OUT_DIR}/{base}_best.pt', map_location=DEVICE)); m.eval()
    for sc, fl in SCEN.items():
        Y, L = predict_logits(m, test_loader_img, **fl)
        Yg, Lg, pat = aggregate(Y, L, test_df)
        SC_P[(base, sc)] = (Yg, sigmoid(Lg), pat)
    del m; torch.cuda.empty_cache()

rows = []
for sc in SCEN:
    r = {'scenario': sc}
    for mdl in ['M7_AGCMF', 'M6_BiLSTM', 'M7_AGCMF_MD', 'M6_BiLSTM_MD']:
        Y, P, pat = SC_P[(mdl, sc)]
        a = mean_auc(Y, P)
        lo, hi = ci(cluster_bootstrap(lambda idx: mean_auc(Y[idx], P[idx]), pat, n_boot=1000))
        r[mdl] = f'{a:.3f} ({lo:.3f}–{hi:.3f})'
    ded = DEDICATED[sc]
    r['dedicated_model'] = ded or '—'
    r['dedicated_AUC'] = f"{SUMM[ded]['mean_AUC']:.3f}" if ded and ded in SUMM else '—'
    # primary comparison: M7_MD vs M6_MD (paired patient bootstrap)
    Y7, P7, pat = SC_P[('M7_AGCMF_MD', sc)]; _, P6, _ = SC_P[('M6_BiLSTM_MD', sc)]
    obs = mean_auc(Y7, P7) - mean_auc(Y7, P6)
    dl = np.array(cluster_bootstrap(lambda idx: mean_auc(Y7[idx], P7[idx]) - mean_auc(Y7[idx], P6[idx]), pat))
    p = max(min(1.0, 2 * min((dl <= 0).mean(), (dl >= 0).mean())), 1.0 / N_BOOT)
    lo, hi = ci(dl)
    r['dAUC_M7MD_vs_M6MD_pp'] = f'{100 * obs:.1f} ({100 * lo:.1f} to {100 * hi:.1f})'
    r['p_raw'] = p
    rows.append(r)
md_table = pd.DataFrame(rows)
md_table['p_Holm'] = multipletests(md_table['p_raw'], method='holm')[1]
md_table.to_csv(f'{OUT_DIR}/table_modality_dropout.csv', index=False)
print('\n=== Modality-dropout experiment: test mean AUC (95% CI), 735 studies ===')
print(md_table.to_string(index=False))

gate_df = pd.DataFrame(gate_rows); gate_df.to_csv(f'{OUT_DIR}/gate_by_scenario_MD.csv', index=False)
print('\n=== AGCMF gate by input condition (does it adapt?) ===')
print(gate_df.to_string(index=False))

# full-input headline metrics for the MD models (val-tuned thresholds)
md_summary = []
for name in MD_MODELS:
    Y, P, pat = SC_P[(name, 'Full trimodal')]
    s = summary_metrics(Y, P, THR[name])
    md_summary.append({'model': name, **{k: round(v, 3) for k, v in s.items()}})
md_summary = pd.DataFrame(md_summary)
print('\n=== Full-input metrics (MD models) ===')
print(md_summary.to_string(index=False))

# external CheXpert for the MD models
ext_md = []
for name in MD_MODELS:
    m = load_md(name)
    Y, L = predict_logits(m, cx_loader, zero_text=True)
    Yg, Pg, pat = cx_aggregate(Y, L)
    adequate = Yg.sum(0) >= 10
    aucs = per_class_auc(Yg, Pg)
    mad = float(np.nanmean(np.where(adequate, aucs, np.nan)))
    lo, hi = ci(cluster_bootstrap(lambda idx: float(np.nanmean(np.where(adequate, per_class_auc(Yg[idx], Pg[idx]), np.nan))),
                                  pat, n_boot=1000))
    ext_md.append({'model': f'{name} (text zeroed)', 'n_studies': len(Yg),
                   'mean_AUC_classes_ge10_pos': round(mad, 3), '95CI': f'{lo:.3f}–{hi:.3f}'})
    del m; torch.cuda.empty_cache()
ext_md = pd.concat([pd.DataFrame(ext_md), ext_df[['model', 'n_studies', 'mean_AUC_classes_ge10_pos', '95CI']]],
                   ignore_index=True)
ext_md.to_csv(f'{OUT_DIR}/external_chexpert_with_MD.csv', index=False)
print('\n=== External CheXpert (no reports) incl. modality-dropout models ===')
print(ext_md.to_string(index=False))

infos = [json.load(open(f'{OUT_DIR}/{n}_info.json')) for n in MD_MODELS]
out = ['# Modality-dropout experiment (pre-specified; all results reported)\n',
       f'p_drop = {P_DROP}; selection = mean(val AUC full, val AUC no-text); same protocol as Step 1\n',
       '## Test mean AUC by input condition', md_table.to_string(index=False),
       '\n## Full-input metrics (MD models)', md_summary.to_string(index=False),
       '\n## AGCMF gate by input condition', gate_df.to_string(index=False),
       '\n## External CheXpert (no reports)', ext_md.to_string(index=False),
       '\n## Training', pd.DataFrame(infos).to_string(index=False)]
with open(f'{OUT_DIR}/REVISION_NUMBERS_MODALITY_DROPOUT.md', 'w') as f:
    f.write('\n'.join(out))
print(f'\nSaved {OUT_DIR}/REVISION_NUMBERS_MODALITY_DROPOUT.md — send this back.')
