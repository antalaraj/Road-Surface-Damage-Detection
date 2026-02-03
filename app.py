import streamlit as st
import zipfile
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern
import shutil
import tempfile

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 0. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Road Damage AI Inspector",
    page_icon="🛣️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛣️ AI Pothole & Road Damage Detector")
st.markdown("Upload a ZIP file of road images to detect cracks, analyze severity, and classify risk levels using Computer Vision & Random Forest.")

# ==========================================
# 1. HELPER FUNCTIONS (CV Logic)
# ==========================================

def preprocess(img):
    """Reads image array, converts to grayscale, and applies adaptive thresholding."""
    if img is None:
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Noise reduction
    gray = cv2.medianBlur(gray, 5)

    # Adaptive Thresholding for Cracks
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return gray, binary

def texture_feature(gray):
    """Calculates Local Binary Pattern (LBP) standard deviation."""
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    return np.std(lbp)

def severity_score(area, compactness, texture):
    """Calculates a custom severity score."""
    return 0.5 * np.log(area + 1) + 0.3 * compactness + 0.2 * texture

# ==========================================
# 2. CORE PROCESSING ENGINE
# ==========================================

@st.cache_data
def process_uploaded_zip(uploaded_zip):
    """
    Extracts zip, processes images, and returns a DataFrame + Dictionary of processed images.
    Cached to prevent re-running on every interaction.
    """
    
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Extract
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    image_files = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    records = []
    processed_images = {} # To store images for display in UI

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, img_path in enumerate(image_files):
        # Update progress
        progress = (idx + 1) / len(image_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing image {idx+1} of {len(image_files)}...")

        # Read Image (OpenCV reads in BGR, convert to RGB for Streamlit)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray, binary = preprocess(img_rgb)

        # Feature Extraction
        texture = texture_feature(gray)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox_img = img_rgb.copy()
        total_area = 0
        max_severity = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400: continue # Filter noise

            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            
            compactness = (peri**2) / (4 * np.pi * area)
            score = severity_score(area, compactness, texture)
            
            max_severity = max(max_severity, score)
            total_area += area

            # Draw on RGB image
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (255, 0, 0), 3) # Red in RGB
            cv2.putText(bbox_img, f"{score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        file_name = os.path.basename(img_path)
        
        # Store data
        records.append([file_name, texture, total_area, max_severity])
        
        # Store images for visualization (Original, Binary Mask, Result)
        processed_images[file_name] = {
            "original": img_rgb,
            "mask": binary,
            "result": bbox_img
        }

    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    status_text.text("Processing Complete!")
    progress_bar.empty()

    df = pd.DataFrame(records, columns=["filename", "texture", "damage_area", "severity_score"])
    
    return df, processed_images

# ==========================================
# 3. SIDEBAR & INPUT
# ==========================================
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Pothole Images (ZIP)", type="zip")

if uploaded_file is not None:
    # RUN PROCESSING
    df, image_data = process_uploaded_zip(uploaded_file)

    # --- ML LOGIC ---
    # Auto-labeling based on quantiles
    if not df.empty and df['damage_area'].sum() > 0:
        df["risk_level"] = pd.qcut(df["damage_area"], 3, labels=["Low", "Medium", "High"])
    else:
        df["risk_level"] = "Low" # Fallback if no damage detected

    # Train Model
    X = df[["texture", "damage_area"]]
    y = df["risk_level"]
    
    # Needs at least 2 samples to split
    if len(df) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
    else:
        acc = 0
        report = {}
        st.sidebar.warning("Not enough images to train ML model reliably (need > 5).")

    # ==========================================
    # 4. DASHBOARD TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Analysis", "🖼️ Image Inspector", "💾 Raw Data"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        # Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Images", len(df))
        c2.metric("Avg Severity", f"{df['severity_score'].mean():.2f}")
        c3.metric("High Risk Roads", len(df[df['risk_level'] == "High"]))
        c4.metric("Model Accuracy", f"{acc*100:.1f}%")

        st.divider()

        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("Region-wise Severity Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df, 
                x="filename", 
                y="damage_area", 
                hue="risk_level", 
                palette={"Low":"green", "Medium":"orange", "High":"red"},
                size="severity_score",
                sizes=(50, 200),
                alpha=0.8,
                ax=ax
            )
            plt.xticks(rotation=90)
            plt.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)

        with col_right:
            st.subheader("Risk Distribution")
            risk_counts = df['risk_level'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['red', 'orange', 'green'])
            st.pyplot(fig2)

    # --- TAB 2: IMAGE INSPECTOR ---
    with tab2:
        st.subheader("Visual Inspection")
        
        # Dropdown to select image
        selected_file = st.selectbox("Select an Image to Inspect:", df['filename'].tolist())
        
        if selected_file:
            row_data = df[df['filename'] == selected_file].iloc[0]
            imgs = image_data[selected_file]
            
            # Show stats for this specific image
            s1, s2, s3 = st.columns(3)
            s1.info(f"Texture Score: {row_data['texture']:.2f}")
            s2.warning(f"Damage Area: {row_data['damage_area']:.0f} px")
            s3.error(f"Predicted Risk: {row_data['risk_level']}")
            
            # Show images side-by-side
            i1, i2, i3 = st.columns(3)
            
            with i1:
                st.image(imgs['original'], caption="Original Image", use_container_width=True)
            with i2:
                st.image(imgs['mask'], caption="Crack Mask (Binary)", use_container_width=True)
            with i3:
                st.image(imgs['result'], caption="AI Detection Result", use_container_width=True)

    # --- TAB 3: RAW DATA ---
    with tab3:
        st.subheader("Processed Dataset")
        st.dataframe(df, use_container_width=True)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Analysis CSV",
            csv,
            "road_damage_report.csv",
            "text/csv",
            key='download-csv'
        )

else:
    # Start Screen
    st.info("👈 Please upload a 'potholes.zip' file in the sidebar to begin analysis.")
    st.markdown("""
    ### How it works:
    1. **Upload** a zip file containing road images.
    2. **Processing**: OpenCV algorithms extract texture and crack contours.
    3. **Scoring**: A custom severity score is calculated based on area and compactness.
    4. **ML Training**: A Random Forest model trains on the extracted features.
    5. **Results**: Visualize the data and inspect individual detection masks.
    """)