# CFHbQ: Image-Based Quantification of Plasma Cell-Free Hemoglobin and Hemolysis Assessment

Code accompanying the manuscript *"Rapid and Non-Destructive Imaging-Based Quantification of
Cell-Free Hemoglobin for Hemolysis Assessment"* (under review).

CFHbQ predicts cell-free hemoglobin (CFHb) concentration from a single photograph of a centrifuged
plasma tube (43 color features → RFECV feature selection → Stacking ensemble → out-of-fold piecewise
recalibration), and classifies hemolysis with a 50 mg/dL decision threshold.

## Repository structure

```
.
├── plasma_pipeline.py            # main pipeline: batch correction → feature extraction →
│                                 #   feature selection → training → OOF recalibration → evaluation
├── model_training_v2_1.py        # model trainer (RF / XGBoost / MLP / Stacking / Voting)
├── cut_plasma_target_region.py   # plasma ROI segmentation (Lab thresholding + morphology)
├── reproduce_model_v4.sh         # one-command reproduction (frozen features, no image processing)
├── requirements.txt              # pinned environment
├── sample_list.txt               # per-image metadata (batch, split, concentration, flags)
├── splitdata_formodel.txt        # batch-level train/test split definition
├── MODEL_CARD.md                 # model card (configuration & official metrics)
└── results/                      # official model outputs (model_V4)
    ├── features_20260818_170937.csv           # frozen 43-feature snapshot (526 × 43)
    ├── calibration.json                       # piecewise recalibration (breakpoint 40 mg/dL)
    ├── correction_factors.csv                 # per-batch RGB correction factors
    ├── stacking_model.pkl                     # deployed Stacking model (load with joblib.load)
    ├── summary.csv, predictions_test_set*.csv # evaluation outputs
    ├── oof_predictions.csv                    # out-of-fold predictions on the 289 training images
    ├── stacking_conc_predict_validation_results.csv  # internal-validation predictions
    │                                          #   (pre-recalibration; see results/calibration.json)
    └── v3.1_official_results.json             # consolidated official metrics
```

The imaging data (`data/blood_imag/`) are not stored in this repository — see **Data** below.

## Data

The imaging dataset is deposited in the Zenodo archive (**DOI: to be added**). Download and extract
it so that the images sit under `data/blood_imag/` in this repository; the full pipeline
(`python plasma_pipeline.py`) will then run from the raw images. Without the images, the quick
reproduction route below still works from the frozen artifacts in `results/`.

* **Plasma images** (`data/blood_imag/<batch>/plasma/<sample_id>/`): one photograph per sample
  (centrifuged EDTA plasma tube; tube kept in a fixed holder against an LED backlight).
  526 images from 25 acquisition batches were used for model development and evaluation;
  the 164-image independent test set comprises the same 41 plasma samples imaged under
  four background-light conditions (bgl1100 / bgl1100_Baffle / bgl1050 / bgl1120).
* **Standard color solutions** (`data/blood_imag/<batch>/calibration/<HUE>/`): Chinese Pharmacopoeia
  (2020) standard colorimetric solutions (cobalt chloride, potassium dichromate and copper sulfate
  mixed in defined proportions, filled in Nessler tubes), six hues (GY, YG, Y, OY, OR, BR) × levels
  0.5–10; levels 5–10 are included here, as used in this study. Batch correction factors are derived
  from these solutions only (reference level estimated from the training batches; no plasma image or
  label enters factor estimation).
* `sample_list.txt` columns: `SampleName`, `BatchID`, `SampleID`, `Concentration` (mg/dL),
  `splitData` (`train_set` / `test_set1–4`), `exclude`, illumination flags, etc.

## Environment

```
pip install -r requirements.txt      # python 3.10; scikit-learn 1.3.0, xgboost 0.82, opencv, skimage, ...
```

## Reproduction

**Quick (recommended, minutes):** reproduces the deployed model from the frozen feature snapshot —
no image processing required.

```bash
bash reproduce_model_v4.sh            # outputs to model_v4_repro/
```

Expected (identical to `results/`): independent test mean R² = **0.9241**, RMSE = 97.5 mg/dL;
internal validation (n = 73) R² = **0.8915**, RMSE = 74.0 mg/dL; hemolysis accuracy 98.8 %
(threshold 50 mg/dL, independent test). Note: the pipeline's console prints the internal-validation
metrics after applying only the high-segment recalibration line (R² = 0.8916, RMSE = 73.97); the
piecewise-recalibrated values reported in the paper (0.8915 / 74.0) are obtained by applying
`results/calibration.json` and are also listed in `results/v3.1_official_results.json`.

**Full (from images):** recomputes features from the raw images in `./data/blood_imag` and reruns
the whole pipeline (feature extraction → RFECV → training → recalibration → evaluation).

```bash
python plasma_pipeline.py                                  # automatic: uses ./data/blood_imag
python plasma_pipeline.py --fixed_features=model_v4        # + skip RFECV (deterministic 15 features)
```

Note: RFECV sits on a flat CV plateau where two features (`Lab_L_median` vs `CIECAM02_J`) differ in
Ridge |coef| by ~1e-12, so its selection can flip under floating-point noise of image processing.
For bit-level reproducibility use `--fixed_features=model_v4` (the 15 official features written in
`plasma_pipeline.py`), which is also the route used by `reproduce_model_v4.sh`.

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Corresponding author (see manuscript). Issues and questions: please open a GitHub issue.
