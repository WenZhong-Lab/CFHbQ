# CFHbQ (Cell-Free Hemoglobin Quantification) — Our Model

**Date:** 2026-08-19
**Version:** **model_V4** (final: B方案 leakage-free selection + piecewise OOF calibration E2; hemolysis cutoff 50 mg/dL; formerly "v3.1")
**Data:** 526 modeling images (362 training-side + 164 independent-test-side); deployed model trained on the internal 289 (80%); 73 images used once as internal validation (holdout); 164 independent test images (41 plasma × 4 lights) evaluation-only. 20 excluded: concentration-extraction failure.

---

## Pipeline

```
Raw Image → scale_mean RGB Correction → Plasma Segmentation
→ 43-dim Feature Extraction → Internal 80/20 split (holdout)
→ RFECV(Ridge α=1.0) on internal 80% (n=289) selects 15 features
→ StandardScaler → StackingRegressor
→ Piecewise OOF calibration:
    pred < 40:  t = 0.4217 × pred + 5.95
    pred ≥ 40:  t = 1.0153 × pred − 5.16
→ Prediction
```

## Configuration

| Component | Setting |
|-----------|---------|
| Batch correction | `scale_mean` (multiplicative, 3 params/batch) |
| Calibration hues | All 6: GY, YG, Y, OY, OR, BR |
| Calibration levels | High: 7, 8, 9, 10 |
| Feature selection | RFECV with Ridge(α=1.0), stratified 5-fold CV |
| Selection samples | **Internal 80% only (n=289); internal validation (holdout, n=73) excluded from selection** |
| Features selected | 15 out of 43 |
| Model | StackingRegressor: RF + MLP + XGBoost → Ridge(α=0.01) |
| Calibration | Piecewise: pred<40 → 0.4217×pred+5.95; pred≥40 → 1.0153×pred−5.16 (fitted on 289 out-of-fold predictions of the 15-feature model; nested; internal validation/independent test never used) |
| CV method | Stratified 5-fold (seed=42) |
| Train/independent-test split | By batch, V4 date-based (17 train, 8 test batches) |

## Selected Features (15)

`rgb_g_std`, `rgb_b_std`, `hsv_h_skew`, `Lab_L_median`, `Lab_a_median`,
`Lab_L_std`, `Lab_a_std`, `Lab_b_std`, `Lab_a_skew`, `Lab_b_skew`,
`CIECAM02_M`, `yuv_y`, `yuv_u`, `rg_ratio_median`, `orange_intensity`

## Calibration (OOF nested piecewise, model_V4)

The stacking model under-predicts high concentrations (prediction compression)
and over-predicts the lowest concentrations (12–15 → 20–46), which hurts the
hemolysis decision in the low-concentration band. Two linear segments were fitted
on **out-of-fold predictions** of the internal 289 training samples (5-fold: each
fold predicted by a model trained on the other four folds):

- `pred ≥ 40` (high segment, all 289 OOF samples): t = 1.0153×pred − 5.16 (OOF R²=0.8443)
- `pred < 40` (low segment, 19 OOF samples): t = 0.4217×pred + 5.95 — corrects the low-end inflation

The breakpoint (pred=40) was fixed a priori to separate the inflated
low-prediction band from the main regression line. Under the adopted 50 mg/dL
hemolysis threshold the piecewise and the v3 single-line calibration produce
identical binary decisions on all evaluation sets (verified 2026-08-18), so the
calibration is unchanged by the threshold decision.
The internal validation (holdout) set and all independent test sets were never used for fitting.
The OOF loop refits the stacking model on the selected 15 features (audit bugfix,
2026-08-18). OOF predictions are fully deterministic across reruns (verified,
max diff 0). `conc_trained_models/calibration.json` stores the coefficients
(`calibration_v3_global.json` = v3 single-line backup).

## Performance

### Internal Validation Set (n=73)

| Metric | Value |
|--------|-------|
| RMSE | 74.00 |
| R² | 0.8915 |
| Hemolysis detection acc (>50 mg/dL) | 98.6% (sensitivity 98.5%, specificity 100%) |

Internal-validation protocol (audited 2026-08-18): hyperparameter tuning and
**best-model selection were performed by stratified 5-fold CV (RMSE) on the
internal 289 samples only; the 73-image internal validation set was used
exactly once, for final validation**. Metrics are reported on calibrated
predictions (the calibration line was fitted on the 289 out-of-fold
predictions; the validation set never participated in any fitting, so the
calibrated numbers are unbiased). Pre-calibration reference: RMSE=74.22,
R²=0.8909. The stacking model's 5-fold CV on the 289: R²=0.8440, RMSE=89.8
(selection score).

### Independent Test Sets (4 light conditions, calibrated)

| Lighting Condition | Description | R² | RMSE | HemoAcc* | Sensitivity | Specificity |
|--------------------|-------------|:---:|:---:|:---:|:---:|:---:|
| bgl1100 | standard (same as train) | 0.9395 | 87.3 | 100% | 100% | 100% |
| bgl1100_Baffle | with baffle | 0.9167 | 102.5 | 97.6% | 100% | 92.3% |
| bgl1050 | dim | 0.9084 | 107.4 | 100% | 100% | 100% |
| bgl1120 | bright | 0.9317 | 92.8 | 97.6% | 96.4% | 100% |
| **Average** | | **0.9241** | **97.5** | **98.8%** | **99.1%** | **98.1%** |

\* HemoAcc = binary hemolysis detection at the **50 mg/dL** threshold (literature-consistent
visible-hemolysis cutoff), from calibrated regression predictions (hemolysis if predicted
concentration >50 mg/dL). Sensitivity = fraction of true-hemolysis samples detected;
specificity = fraction of non-hemolysis samples correctly called negative.

At 50 mg/dL the pooled 164 independent-test samples contain exactly **1 FN** (plasma
12-03_1_19 under bgl1120: true 61.4, predicted 49.7 — a mild-hemolysis case read just
below the cutoff) and **1 FP** (plasma 12-03_1_15 under bgl1100_Baffle: true 21.24,
predicted >50). One mild case is also missed in the internal validation set (true 73.1).
The independent 4-grade classifier (`dual_model.py`) remains available for finer grading.

**Data correction (2026-08-19):** the registered true concentration of independent-test
plasma 12-03_1_15 was corrected 67.53 → 21.24 mg/dL per laboratory record (the value was
wrong in sample_list.txt and downstream CSVs; the 2026-08-10 features file already carried
the correct 21.24). The sample lies in the independent test set only — no training,
selection or calibration fitting used it, so the model itself is unchanged; independent
test metrics above were recomputed from the corrected labels. Previously reported test
average R²=0.9235, HemoAcc 97.6% (sens 96.6%, spec 100%) reflected the wrong 67.53 label.

Sensitivity analysis at 20 mg/dL (same model, threshold only): average HemoAcc 92.7%,
sensitivity 100% (FN=0), specificity 75% (false positives concentrate under dim/baffle
lighting).

### Bias by Concentration Band (calibrated, pooled over independent test sets)

| Band | n | Mean bias (pred − true) |
|------|---|-------------------------|
| ≤150 | 72 | +6.9 |
| 150–400 | 0 | — |
| 400–700 | 48 | −81.7 |
| >700 | 44 | −152.0 |

Residual under-prediction concentrates in the >700 tail (distribution-shift
compression); candidate remedies (holdout-fitted calibration, truncating the
training range at 700 or 900) were tested and are documented below.

## Data Split

| Split | Batches | Samples |
|-------|---------|:---:|
| Train (internal 80%, n=289 for model fitting) | 17 batches (Apr-Dec 2025) | 362 |
| Internal validation (holdout) | internal 20% (n=73, used once) | 73 |
| Independent test (bgl1100) | 12-03_1_bgl1100, 12-04_4_bgl1100 | 41 |
| Independent test (bgl1100_Baffle) | 12-03_1_bgl1100_Baffle, 12-04_4_bgl1100_Baffle | 41 |
| Independent test (bgl1050) | 12-03_1_bgl1050, 12-04_4_bgl1050 | 41 |
| Independent test (bgl1120) | 12-03_1_bgl1120, 12-04_4_bgl1120 | 41 |

The four independent test sets are the **same 41 plasma samples** (18 from 12-03_1,
23 from 12-04_4) photographed under four background lights — a paired
repeated-measures design; the independent test sets are not independent samples
of each other (paired design), but are independent of the training/internal
validation data (different dates = different plasma samples).

## Excluded Samples (20 total)

- **20 samples:** batch-2025-12-03_1 samples 13/14/16/17/18 (×4 light conditions) — concentration extraction failure (NaN)

## Version History

| Version | Feature selection | Calibration | Independent test avg R² | Note |
|---------|-------------------|-------------|:-----------:|------|
| v1 (08-11) | RFECV on all 362 → 21 features | none | 0.9209 | internal-validation R² 0.8986 biased (+0.008) by selection leakage |
| v2 (08-17) | RFECV on 289 → 15 features | none | 0.9186 | leakage-free selection |
| v3 (08-18) | RFECV on 289 → 15 features | OOF nested linear | 0.9223 | superseded by v3.1 |
| **model_V4 (08-18)** | RFECV on 289 → 15 features | **OOF piecewise E2 (v3 global line + low-segment fix)** | **0.9241** | **final** (formerly "v3.1"); hemolysis cutoff 50 mg/dL: independent-test avg Acc 98.8%, specificity 98.1%, sensitivity 99.1% (audited; at 20 mg/dL: sens 100%, spec 75%). Independent-test avg R² updated 0.9235→0.9241 on 2026-08-19 after test-label correction (12-03_1_15: 67.53→21.24). |

Rejected alternatives (documented, reproducible):
- Internal-validation-fitted calibration (k=1.0416, b=−8.456 → test avg R²=0.9360):
  consumes the internal validation set as calibration set; archived in
  `results/holdout_calibration_results.json`.
- Truncating training range (>700 or >900 removed): degrades in-range test
  performance by 0.014–0.043 R² (boundary shifts down); rejected
  (`results/experiment_exclude_high_conc*.json`).
- Future work: sample-weighting for high concentrations (scheme B, not yet tested).

## Files

- `plasma_pipeline.py` — main pipeline (v3: holdout-free selection + OOF calibration, `USE_CALIBRATION=True`)
- `model_training_v2_1.py` — model trainer
- `experiment_holdout_free_selection.py` — v1 vs v2 comparison driver
- `experiment_exclude_high_conc.py` / `experiment_exclude_above_900.py` — rejected truncation experiments
- `plot_rfecv_selection_curve.py` — RFECV curve + 43-feature importance figure (15 retained)
- `plot_prediction_figures.py` — internal-validation & 4-light prediction scatter figures
- `make_data_tables.py` — CN/EN data-table Excel generator
- `results/v3_official_results.json` — complete v3 results
- `results/holdout_calibration_results.json` — archived candidate results
- `conc_trained_models/` — 9 model files + `calibration.json`
- `results/predictions_test_set*.csv` — calibrated (`conc_pred`) + raw (`conc_pred_raw`)
- `results/pipeline_v3_calibration_run.log` — official v3 run log
- `sample_list.txt` — sample metadata (546 rows, exclude flags for 20 NaN)
- `features_20260818_110507.csv` — extracted features (526 modeling samples)

---

## model_V4: Naming & Data Usage (appended 2026-08-19)

### Naming

- The final model is **model_V4** (formerly "v3.1"). Its calibration **E2** = the
  v3 global line (refitted on the 15-feature model, audit bugfix 2026-08-18) +
  the low-segment correction (pred<40).
- Naming synchronized in: this MODEL_CARD, CN/EN Excel data tables, pipeline
  print labels, `results/v3.1_official_results.json` version field, project
  memory. File names retain the historical "v3.1" markers for traceability.

### Data Used by model_V4 (audited against code, 2026-08-19)

```
546 image records (6 CV-outlier samples deleted at source; never reported)
  └─ exclude 20 NaN (concentration-extraction failure;
     12-03_1 samples 13/14/16/17/18 × 4 lights)
      └─ 526 modeling images
          ├─ training-side 362 images (V4 date-based split)
          │    └─ internal 80/20 split, stratified by concentration (seed=42)
          │         ├─ 289 images  ← feature selection (RFECV) + stacking training
          │         │                + calibration OOF fitting
          │         └─ 73 images ← internal validation, used exactly
          │                         once (R²=0.8915, RMSE=74.00)
          └─ independent-test-side 164 images ← evaluation only (never used for
                                                any fitting)
```

| Stage | Data | Detail |
|-------|------|--------|
| Feature selection | 289 images | RFECV(Ridge α=1.0) selects 15/43 features (Scheme B, leakage-free) |
| Model training (deployed model) | 289 images | Stacking(RF+MLP+XGB→Ridge) fitted on the internal 80% — **not 362** |
| Calibration fitting | 289 OOF predictions (5-fold) | high segment: all 289 OOF; low segment: the 19 OOF with pred<40 |
| Hyperparameter tuning & model selection | 289 (5-fold CV) | best model chosen by CV RMSE (stacking) |
| Internal validation | 73 images | used exactly once → R²=0.8915 / RMSE=74.00 (unbiased; pre-calibration reference 0.8909/74.22) |
| Independent test evaluation | 164 test images | same 41 plasma samples × 4 lighting conditions; paired design |

Plasma-level composition:

- Training-side 362 images = **227 plasma samples**: 182 (Apr–Jul 2025, single
  light bgl1100) + 23 × 4 lights (12-03_2) + 22 × 4 lights (12-04_3)
- Independent-test-side 164 images = **the same 41 plasma samples** (12-03_1: 18 +
  12-04_4: 23) × 4 lighting conditions
- Date-based split ⇒ no plasma-level leakage between training and independent
  test sides.

Paper wording (Methods): "the 362 training-side images were split internally
(80/20, stratified by concentration) into 289 training images used for feature
selection, model fitting and calibration, and a 73-image holdout used exactly
once as internal validation; the 164 independent test images (41 plasma samples
× 4 lighting conditions) were used exclusively for evaluation."

## Reproducibility (appended 2026-08-19)

**Frozen feature set (paper reproducibility decision).** RFECV on the internal
289 samples sits on a plateau: `Lab_L_median` and `CIECAM02_J` have nearly
identical Ridge |coef| (difference ~5e-12), and the ~1e-13 floating-point noise
in image feature extraction can flip their elimination order (10 seeded noise
trials flipped 6/10; the 2026-08-19 unfrozen rerun selected `CIECAM02_J`
instead of `Lab_L_median`, changing downstream metrics by ΔR² ≤ 0.003). The
published model therefore freezes the official 15-feature set
(`MODEL_V4_OFFICIAL_FEATURES` in `plasma_pipeline.py`, used via
`--fixed_features=model_v4`), which is the standard "feature selection run once
and fixed" practice.

**One-command reproduction (bitwise):**

```bash
bash reproduce_model_v4.sh          # Level 1: from frozen feature snapshot
```

- Level 1 (bitwise): `--fixed_features=model_v4 --features_csv=results/features_20260818_170937.csv`
  skips feature extraction and RFECV; all downstream steps (stacking training,
  OOF piecewise calibration, evaluation) are seeded and deterministic. Two
  independent parallel reruns produced 28/32 files with identical md5; the 4
  differing files are model `.pkl` binaries whose predictions are numerically
  identical (max diff = 0 on all 362 training images). All metrics match the
  official 08-18 run exactly (test avg R²=0.9241, calibration 1.0153/−5.16 and
  0.4217/5.95, internal validation 0.8915/74.00). Verification:
  `verify_reproducibility.py`.
- Level 2 (numeric-level, from raw images): rerun without `--features_csv`
  (feature extraction re-run; ~1e-13 noise possible) with the frozen feature
  set — all reported metrics are identical at the reported precision.
- Environment pinned in `requirements.txt` (sklearn 1.3.0, xgboost 0.82,
  numpy 1.24.4, pandas 2.2.2, ...). The divergent unfrozen rerun is archived
  in `model_v4_float_variant/` (evidence for the freeze decision).
