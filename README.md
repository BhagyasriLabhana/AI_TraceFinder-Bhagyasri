
# TraceFinder — Scanner Identification & Tamper Detection

This project identifies the source scanner of a document image and detects tampering by analyzing residual device artifacts and texture/frequency signatures, supporting forensic investigations, document authentication, and legal verification.[2][1]

## Objectives

- Identify the scanner brand/model from intrinsic noise, texture, and compression traces learned by ML models.[1]
- Detect manipulations such as copy‑move, retouch, and splicing in scanned images.[2]

## Use cases

- Digital forensics: attribute forged or duplicated documents to specific scanners.[1]
- Document authentication: differentiate authorized vs unauthorized scanner outputs.[1]
- Legal verification: confirm scans originate from approved devices.[1]

## System overview

- Residual preprocessing using Haar DWT denoising to enhance device/tamper signals.[2]
- Scanner ID: hybrid CNN + handcrafted 27‑D features (11 correlations to fingerprints + 6 FFT radial energies + 10 LBP‑uniform).[2]
- Tamper detection:
  - Image‑level 18‑D per‑patch features (10 LBP + 6 FFT + 2 contrast stats [std, mean(|x|−mean)]), averaged across patches and classified via calibrated SVM. [2]
  - Patch‑level 22‑D fallback (10 LBP + 6 FFT + 3 residual stats + 3 FFT resample stats) with top‑k aggregation.[2]

## Milestones and timeline

### Milestone 1 — Dataset & preprocessing (Weeks 1–2)
- Collect scanned samples from 3–5+ scanner models; create labels (scanner_model, file_name, page_id).[1]
- Analyze resolutions, formats, channels; normalize folder structure and manifests.[1]
- Preprocess: grayscale, resize to 256×256, Haar DWT denoise (zero cH/cV/cD), residual = img − denoise.[2][1]
- Outputs:
  - Labeled dataset and manifest CSVs.[1]
  - Verified residual generation pipeline and example residual images.[2]

### Milestone 2 — Feature engineering & baselines (Weeks 3–4)
- Extract handcrafted features:
  - FFT radial band energies, LBP‑uniform histograms, residual statistics, PRNU/noise maps (optional).[1]
- Train baseline models (Logistic Regression, SVM, Random Forest) and evaluate accuracy/confusion matrix.[1]
- For tamper: implement patch descriptors (22‑D) and image‑level descriptors (18‑D) as defined in Colab.[2]
- Outputs:
  - Baseline metrics (accuracy, confusion matrix).[1]
  - Visuals of noise/residual maps across scanners.[1]

### Milestone 3 — Deep model + explainability (Weeks 5–6)
- Train hybrid CNN for scanner ID with dual inputs: residual image and 27‑D handcrafted vector; apply augmentation.[2][1]
- Evaluate (accuracy, F1, confusion matrix) and apply explainability (Grad‑CAM/SHAP) to highlight scanner‑specific patterns.[1]
- Outputs:
  - Keras model (scanner_hybrid.keras), scaler (27‑D), label encoder, reference fingerprints and key order.[2]
  - Explainability visuals and validation metrics.[1]

### Milestone 4 — Deployment & reporting (Weeks 7–8)
- Streamlit app:
  - Upload image → residual + 27‑D features → hybrid model predicts scanner.[2]
  - Tamper detection prefers image‑level (18‑D patch‑avg) with calibrated SVM; falls back to patch‑level (22‑D).[2]
  - Thresholding caps per‑domain threshold to the global value to avoid over‑strict tamper_dir settings.[2]
- Final documentation:
  - System architecture, training results, model comparisons, screenshots of app outputs.[1]
- Outputs:
  - Deployed app.py and all artifacts, installation instructions, demo screenshots.[2][1]

## Methods

### Residual preprocessing
- Grayscale → resize $$256×256$$ → Haar DWT denoise by zeroing detail bands → inverse DWT → residual = image − denoise.[2][1]

### Scanner identification
- 27‑D handcrafted vector:
  - 11× corr2d with stored scanner fingerprints (fp_keys order), 6× FFT radial energies, 10× LBP‑uniform.[2]
- Dual‑input Keras model consumes residual image + standardized 27‑D vector; label via hybrid_label_encoder.pkl.[2]

### Tamper detection
- Image‑level (preferred):
  - Per patch 18‑D = 10 LBP + 6 FFT + 2 contrast stats [std, mean(|x| − mean)], averaged across MAX_PATCHES. [2]
  - Standardize with image_scaler.pkl; calibrated SVM outputs probability; threshold uses min(domain, global).[2]
- Patch‑level (fallback):
  - Per patch 22‑D = 10 LBP + 6 FFT + 3 residual stats + 3 FFT resample stats; calibrated SVM; top‑k mean aggregation with local‑hit gating.[2]

## Installation

- Python 3.10+ recommended. Install dependencies:
  - pip install -r requirements.txt[2]
- Place artifacts next to app.py or update paths:
  - Scanner: scanner_hybrid.keras, hybrid_label_encoder.pkl, hybrid_feat_scaler.pkl, scanner_fingerprints.pkl, fp_keys.npy.[2]
  - Tamper image‑level: image_scaler.pkl, image_svm_sig.pkl, image_thresholds.json.[2]
  - Tamper patch‑level: patch_scaler.pkl, patch_svm_sig_calibrated.pkl, thresholds_patch.json.[2]

## Running the app

- streamlit run app.py and open the shown local URL.[2]
- Upload TIFF/PNG/JPG; the app shows:
  - Scanner label + confidence.[2]
  - Tamper label + probability, threshold used, and debug info (domain, scaler n_features_in_).[2]

## Datasets

- Target: scans from multiple scanners (3–5+ models), balanced across classes.[1]
- Suggested sources: in‑house scans per guidelines; external public datasets (e.g., Kaggle) for augmentation with care.[1]

## Evaluation

- Scanner ID: accuracy, precision, recall, F1, confusion matrix; robustness across resolution/format.[1]
- Tamper: ROC‑AUC, operating thresholds via Youden J; validate domain thresholds and apply capping during deployment.[1][2]

## Results snapshot

- Patch‑level ROC‑AUC around 0.84 observed in Colab validation experiments.[2]
- Image‑level thresholds tuned per domain/type; deployment uses capped domain thresholds to align decisions.[2]

## Repository layout

- app.py — Streamlit app entrypoint.[2]
- requirements.txt — Dependencies.[2]
- AI_TraceFinder-FSI.ipynb — Colab notebook with training and artifact export.[2]
- Artifacts — model and scaler files listed above.[2]
- Bhagyasri/ — workspace folder for submissions, logs, and screenshots.[1]

## Milestone deliverables checklist

- M1: Dataset manifest, preprocessing scripts, residual visualizations.[1]
- M2: Feature extractors, baseline models, metrics plots.[1]
- M3: Hybrid CNN model, explainability reports, validation metrics.[1]
- M4: Streamlit app, deployment artifacts, final documentation and screenshots.[1][2]

## Troubleshooting

- Clean outputs for tampered images:
  - Confirm image_thresholds.json is loaded; check sidebar for img_prob vs img_thr_raw vs img_thr_used; ensure threshold cap is active.[2]
- Feature dimension errors:
  - Scanner scaler n_features_in_ = 27; image scaler n_features_in_ = 18; patch scaler expects 22 per patch.[2]

## License

- MIT license; follow dataset licenses and attribute upstream sources.[1]

## Acknowledgments

- Upstream mentor repository and community resources for baseline structure and guidance.[1]

