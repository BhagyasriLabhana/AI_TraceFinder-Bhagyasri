# 📄 Scanner Brand Identifier  

A machine learning and deep learning–based system to identify the **source scanner device** used to scan a document or image. The model leverages **noise patterns, frequency-domain features, and deep CNNs** to detect the unique artifacts introduced by different scanner brands/models.  

## 🎯 Objective  
The aim of this project is to identify the scanner brand/model used for digitizing a document by analyzing unique traces (noise, texture, compression artifacts) left during the scanning process. Each scanner introduces distinct patterns that can be captured using feature extraction and learned by ML/DL models.  

## 🚀 Features  
- Upload a single/multiple scanned images or a ZIP file  
- Predict the scanner brand/model with **confidence scores**  
- Logs past predictions with thumbnails  
- Download results as CSV for reporting  
- Hybrid model: combines **CNN (deep residual features)** + **hand-crafted features (PRNU, FFT, LBP, correlations)**  

## 📂 Repository Structure  
```
Bhagyasri/
├── ScannerIdentification.ipynb       # Colab notebook: data collection, experiments, training
├── Scanner-Brand-Identifier/         # Deployment & model artifacts
│ ├── streamlit_app.py                # Streamlit    app UI for predictions
│ ├── scanner_fingerprints.pkl        # Saved fingerprint patterns extracted during preprocessing
│ ├── scanner_hybrid.keras            # Trained hybrid CNN + feature model
│ ├── hybrid_label_encoder.pkl        # Label encoder for scanner brands
│ ├── hybrid_feat_scaler.pkl 	       # Scaler for hand-crafted features
│ ├── fp_keys.npy		       # Order mapping for fingerprint features
│ └── README.md 		       # README for the deployment folder
└── README.md                         # Main project documentation (this file)

## ⚙️ Setup & Installation  

### 🔹 Local Setup  
1. Clone this repository:  
   ```bash
   git clone https://github.com/BhagyasriLabhana/AI_TraceFinder-Bhagyasri.git
   cd AI_TraceFinder-Bhagyasri/Bhagyasri/Scanner-Brand-Identifier
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit app:  
   ```bash
   streamlit run streamlit_app.py
   ```

### 🔹 Colab Notebook  
- Open ScannerIdentification.ipynb via Google Colab.
- You can use that for data preprocessing, model training, evaluation, and exporting artifacts. 

## 📊 Project Milestones  

### **Milestone 1: Dataset Collection & Preprocessing**  
- Collect scanned samples from multiple devices  
- Resize, normalize, and preprocess images  

### **Milestone 2: Feature Engineering & Modeling
- Extracted key features: PRNU, FFT, LBP, correlations
- Combined these handcrafted features with CNN-based deep features
- Developed a Hybrid Model to improve scanner brand identification accuracy

### **Milestone 3: Deep Learning Model + Explainability**  
- Train CNN-based classifier on raw images  
- Apply Grad-CAM/SHAP for explainability  

### **Milestone 4: Deployment & Reporting**  
- Build Streamlit UI for predictions  
- Enable logging & CSV export  

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **TensorFlow / Keras** (deep learning)  
- **scikit-learn** (classical ML)  
- **OpenCV / PyWavelets** (image preprocessing, wavelet filtering)  
- **Streamlit** (UI deployment)  
- **Pandas / NumPy** (data handling)  

## 📜 Future Work  
- Expand dataset with more scanner brands  
- Improve explainability of model predictions  
- Integrate into forensic analysis pipelines  

Dataset Drive link :- https://drive.google.com/drive/folders/1wEJl8WU29h07RZRutTa_yglpf0jMzmGq?usp=drive_link
