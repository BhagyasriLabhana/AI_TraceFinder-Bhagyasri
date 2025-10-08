
# TraceFinder — Scanner Identification & Tamper Detection

This project identifies the source scanner of a document image and detects tampering by analyzing residual device artifacts and texture/frequency signatures, supporting forensic investigations, document authentication, and legal verification.

## Objectives

- Identify the scanner brand/model from intrinsic noise, texture, and compression traces learned by ML models.
- Detect manipulations such as copy‑move, retouch, and splicing in scanned images.

## Use cases

- Digital forensics: attribute forged or duplicated documents to specific scanners.
- Document authentication: differentiate authorized vs unauthorized scanner outputs.
- Legal verification: confirm scans originate from approved devices.[1]

## System overview

- Residual preprocessing using Haar DWT denoising to enhance device/tamper signals.
- Scanner ID: hybrid CNN + handcrafted 27‑D features (11 correlations to fingerprints + 6 FFT radial energies + 10 LBP‑uniform).
- Tamper detection:
  - Image‑level 18‑D per‑patch features (10 LBP + 6 FFT + 2 contrast stats [std, mean(|x|−mean)]), averaged across patches and classified via calibrated SVM.
  - Patch‑level 22‑D fallback (10 LBP + 6 FFT + 3 residual stats + 3 FFT resample stats) with top‑k aggregation.

## Milestones and timeline

### Milestone 1 — Dataset & preprocessing (Weeks 1–2)
- Collect scanned samples from 3–5+ scanner models; create labels (scanner_model, file_name, page_id).
- Analyze resolutions, formats, channels; normalize folder structure and manifests.
- Preprocess: grayscale, resize to 256×256, Haar DWT denoise (zero cH/cV/cD), residual = img − denoise.
- Outputs:
  - Labeled dataset and manifest CSVs.
  - Verified residual generation pipeline and example residual images.

### Milestone 2 — Feature engineering & baselines (Weeks 3–4)
- Extract handcrafted features:
  - FFT radial band energies, LBP‑uniform histograms, residual statistics, PRNU/noise maps (optional).
- Train baseline models (Logistic Regression, SVM, Random Forest) and evaluate accuracy/confusion matrix.
- For tamper: implement patch descriptors (22‑D) and image‑level descriptors (18‑D) as defined in Colab.
- Outputs:
  - Baseline metrics (accuracy, confusion matrix).
  - Visuals of noise/residual maps across scanners.

### Milestone 3 — Deep model + explainability (Weeks 5–6)
- Train hybrid CNN for scanner ID with dual inputs: residual image and 27‑D handcrafted vector; apply augmentation.
- Evaluate (accuracy, F1, confusion matrix) and apply explainability (Grad‑CAM/SHAP) to highlight scanner‑specific patterns.
- Outputs:
  - Keras model (scanner_hybrid.keras), scaler (27‑D), label encoder, reference fingerprints and key order.
  - Explainability visuals and validation metrics.

### Milestone 4 — Deployment & reporting (Weeks 7–8)
- Streamlit app:
  - Upload image → residual + 27‑D features → hybrid model predicts scanner.
  - Tamper detection prefers image‑level (18‑D patch‑avg) with calibrated SVM; falls back to patch‑level (22‑D).
  - Thresholding caps per‑domain threshold to the global value to avoid over‑strict tamper_dir settings.
- Final documentation:
  - System architecture, training results, model comparisons, screenshots of app outputs.
- Outputs:
  - Deployed app.py and all artifacts, installation instructions, demo screenshots.

## Methods

### Residual preprocessing
- Grayscale → resize $$256×256$$ → Haar DWT denoise by zeroing detail bands → inverse DWT → residual = image − denoise.

### Scanner identification
- 27‑D handcrafted vector:
  - 11× corr2d with stored scanner fingerprints (fp_keys order), 6× FFT radial energies, 10× LBP‑uniform.
- Dual‑input Keras model consumes residual image + standardized 27‑D vector; label via hybrid_label_encoder.pkl.

### Tamper detection
- Image‑level (preferred):
  - Per patch 18‑D = 10 LBP + 6 FFT + 2 contrast stats [std, mean(|x| − mean)], averaged across MAX_PATCHES.
  - Standardize with image_scaler.pkl; calibrated SVM outputs probability; threshold uses min(domain, global).
- Patch‑level (fallback):
  - Per patch 22‑D = 10 LBP + 6 FFT + 3 residual stats + 3 FFT resample stats; calibrated SVM; top‑k mean aggregation with local‑hit gating.

## Installation

- Python 3.10+ recommended. Install dependencies:
  - pip install -r requirements.txt
- Place artifacts next to app.py or update paths:
  - Scanner: scanner_hybrid.keras, hybrid_label_encoder.pkl, hybrid_feat_scaler.pkl, scanner_fingerprints.pkl, fp_keys.npy.
  - Tamper image‑level: image_scaler.pkl, image_svm_sig.pkl, image_thresholds.json.
  - Tamper patch‑level: patch_scaler.pkl, patch_svm_sig_calibrated.pkl, thresholds_patch.json.

## Running the app

- streamlit run app.py and open the shown local URL.
- Upload TIFF/PNG/JPG; the app shows:
  - Scanner label + confidence.
  - Tamper label + probability, threshold used, and debug info (domain, scaler n_features_in_).

## Datasets

- Target: scans from multiple scanners (3–5+ models), balanced across classes.
- Suggested sources: in‑house scans per guidelines; external public datasets (e.g., Kaggle) for augmentation with care.

## Evaluation

- Scanner ID: accuracy, precision, recall, F1, confusion matrix; robustness across resolution/format.
- Tamper: ROC‑AUC, operating thresholds via Youden J; validate domain thresholds and apply capping during deployment.

## Results snapshot

- Patch‑level ROC‑AUC around 0.84 observed in Colab validation experiments.
- Image‑level thresholds tuned per domain/type; deployment uses capped domain thresholds to align decisions.

## Repository layout

- app.py — Streamlit app entrypoint.
- requirements.txt — Dependencies.
- AI_TraceFinder-FSI.ipynb — Colab notebook with training and artifact export.
- Artifacts — model and scaler files listed above.
- Bhagyasri/ — workspace folder for submissions, logs, and screenshots.

## Milestone deliverables checklist

- M1: Dataset manifest, preprocessing scripts, residual visualizations.[1]
- M2: Feature extractors, baseline models, metrics plots.[1]
- M3: Hybrid CNN model, explainability reports, validation metrics.[1]
- M4: Streamlit app, deployment artifacts, final documentation and screenshots.[1][2]

## Troubleshooting

- Clean outputs for tampered images:
  - Confirm image_thresholds.json is loaded; check sidebar for img_prob vs img_thr_raw vs img_thr_used; ensure threshold cap is active.
- Feature dimension errors:
  - Scanner scaler n_features_in_ = 27; image scaler n_features_in_ = 18; patch scaler expects 22 per patch.

## License

- MIT license; follow dataset licenses and attribute upstream sources.

## Acknowledgments

- Upstream mentor repository and community resources for baseline structure and guidance.

