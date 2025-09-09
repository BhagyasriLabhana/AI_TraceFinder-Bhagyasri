import streamlit as st
import os, pickle, numpy as np, tensorflow as tf, cv2, pywt, tempfile, io, base64
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script

FP_PATH = os.path.join(BASE_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(BASE_DIR, "fp_keys.npy")
CKPT_PATH = os.path.join(BASE_DIR, "scanner_hybrid.keras")
LE_PATH = os.path.join(BASE_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "hybrid_feat_scaler.pkl")

# --------------------------
# Cached Loaders
# --------------------------
@st.cache_resource
def load_model(ckpt_path):
    return tf.keras.models.load_model(ckpt_path, compile=False)

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_numpy(path):
    return np.load(path, allow_pickle=True).tolist()

# --------------------------
# Load model + artifacts (cached)
# --------------------------
hyb_model = load_model(CKPT_PATH)
le_inf = load_pickle(LE_PATH)
scaler_inf = load_pickle(SCALER_PATH)
scanner_fps_inf = load_pickle(FP_PATH)
fp_keys_inf = load_numpy(ORDER_NPY)
IMG_SIZE = (256, 256)


# --------------------------
# Utilities
# --------------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = r.max()+1e-6
    bins = np.linspace(0, rmax, K+1)
    feats=[]
    for i in range(K):
        m=(r>=bins[i]) & (r<bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng=float(np.ptp(img))
    if rng<1e-12:
        g=np.zeros_like(img,dtype=np.float32)
    else:
        g=(img-float(np.min(img)))/(rng+1e-8)
    g8=(g*255.0).astype(np.uint8)
    codes=local_binary_pattern(g8,P=P,R=R,method="uniform")
    n_bins=P+2
    hist,_=np.histogram(codes,bins=np.arange(n_bins+1),density=True)
    return hist.astype(np.float32).tolist()

def preprocess_residual_pywt(path):
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,IMG_SIZE,interpolation=cv2.INTER_AREA)
    img=img.astype(np.float32)/255.0
    cA,(cH,cV,cD)=pywt.dwt2(img,'haar')
    cH.fill(0);cV.fill(0);cD.fill(0)
    den=pywt.idwt2((cA,(cH,cV,cD)),'haar')
    return (img-den).astype(np.float32)

def make_feats_from_res(res):
    v_corr=[corr2d(res,scanner_fps_inf[k]) for k in fp_keys_inf]
    v_fft=fft_radial_energy(res,K=6)
    v_lbp=lbp_hist_safe(res,P=8,R=1.0)
    v=np.array(v_corr+v_fft+v_lbp,dtype=np.float32).reshape(1,-1)
    v=scaler_inf.transform(v)
    return v

def predict_scanner_hybrid(image_path):
    res=preprocess_residual_pywt(image_path)
    x_img=np.expand_dims(res,axis=(0,-1))
    x_ft=make_feats_from_res(res)
    prob=hyb_model.predict([x_img,x_ft],verbose=0)
    idx=int(np.argmax(prob))
    label=le_inf.classes_[idx]
    conf=float(prob[0,idx]*100.0)
    return label,conf

# Convert image to base64 for logs
def image_to_base64(img: Image.Image):
    buf=io.BytesIO()
    img.save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Scanner Brand Identifier", layout="wide")
st.title("📄 Scanner Brand Identifier")
st.markdown("Upload scanned images (TIFF/PNG/JPG). The model predicts **scanner brand (HP, Canon, etc.)** with confidence scores.")

# Always reset logs on restart
st.session_state.logs = []

# Tabs
tabs=st.tabs(["🔹 Single Image","🔹 Multiple Images","📂 Upload ZIP","📜 Logs"])

# --- Single Image ---
with tabs[0]:
    uploaded_file=st.file_uploader("Upload a scanned image", type=["png","jpg","jpeg","tif","tiff"], key="single")
    if uploaded_file:
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png")
        tmp.write(uploaded_file.read()); tmp.close()
        label,conf=predict_scanner_hybrid(tmp.name)
        image=Image.open(tmp.name).convert("RGB")
        st.image(
            image,
            caption=f"Prediction: Scanner Model = {label} ({conf:.2f}%)",
            use_container_width=True
        )
        thumb=image.copy(); thumb.thumbnail((96,96))
        st.session_state.logs.append([uploaded_file.name,label,conf,image_to_base64(thumb)])

# --- Multiple Images ---
with tabs[1]:
    uploaded_files=st.file_uploader("Upload multiple scanned images", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True, key="multi")
    if uploaded_files:
        results=[]
        for uf in uploaded_files:
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png")
            tmp.write(uf.read()); tmp.close()
            label,conf=predict_scanner_hybrid(tmp.name)
            results.append([uf.name,label,f"{conf:.2f}%"])
            img=Image.open(tmp.name).convert("RGB")
            thumb=img.copy(); thumb.thumbnail((96,96))
            st.session_state.logs.append([uf.name,label,conf,image_to_base64(thumb)])
        st.dataframe(pd.DataFrame(results,columns=["Filename","Scanner Model","Confidence"]),use_container_width=True)

# --- ZIP Upload ---
with tabs[2]:
    zip_file=st.file_uploader("Upload a ZIP file of scanned images",type=["zip"],key="zip")
    if zip_file:
        tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".zip")
        tmp.write(zip_file.read()); tmp.close()
        results=[]
        with ZipFile(tmp.name,"r") as zip_ref:
            for fname in zip_ref.namelist():
                if fname.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
                    fpath=zip_ref.extract(fname,tempfile.gettempdir())
                    label,conf=predict_scanner_hybrid(fpath)
                    results.append([fname,label,f"{conf:.2f}%"])
                    img=Image.open(fpath).convert("RGB")
                    thumb=img.copy(); thumb.thumbnail((96,96))
                    st.session_state.logs.append([fname,label,conf,image_to_base64(thumb)])
        st.dataframe(pd.DataFrame(results,columns=["Filename","Scanner Model","Confidence"]),use_container_width=True)

# --- Logs ---
with tabs[3]:
    st.subheader("Prediction Logs")
    if st.session_state.logs:
        # DataFrame with HTML thumbnails
        df_logs = pd.DataFrame(
            [
                {
                    "Thumbnail": f'<img src="data:image/png;base64,{thumb}" width="48">',
                    "Filename": f,
                    "Scanner Model": l,
                    "Confidence": f"{c:.2f}%"
                }
                for f, l, c, thumb in st.session_state.logs
            ]
        )
        st.write(df_logs.to_html(escape=False, index=False), unsafe_allow_html=True)

        # CSV (without images)
        df_csv = pd.DataFrame(
            [[f, l, round(c, 2)] for f, l, c, _ in st.session_state.logs],
            columns=["Filename", "Scanner Model", "Confidence"]
        )
        csv = df_csv.to_csv(index=False).encode("utf-8")
        st.download_button("Download Logs (CSV)", data=csv,
                           file_name="scanner_predictions.csv", mime="text/csv")
    else:
        st.info("No logs yet. Upload an image to start logging predictions.")
