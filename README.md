# 🛣️ Road Surface Damage Detection

AI-based system for detecting potholes and cracks on road surfaces using Computer Vision and Machine Learning with an interactive Streamlit dashboard.

---

## 📌 Overview

This project automatically analyzes road images to:
- Detect potholes and cracks
- Calculate damage severity
- Classify roads into risk levels (Low / Medium / High)
- Visualize results using an interactive web dashboard

It is designed as a smart city solution for automated road infrastructure monitoring.

---

## 🚀 Features

- Upload road images as a ZIP file
- Crack segmentation using OpenCV
- Texture analysis using Local Binary Pattern (LBP)
- Severity scoring based on area, shape, and texture
- Risk level prediction using Random Forest
- Interactive data visualization with Streamlit

---

## 🧠 Technology Stack

- Python  
- OpenCV  
- NumPy  
- Pandas  
- Scikit-learn  
- Scikit-image  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## 📁 Project Structure

Road-Surface-Damage-Detection/
│
├── app.py  
├── potholes.zip  
└── README.md   

---

## ⚙️ Installation & Run

1. Clone the repository
git clone https://github.com/antalaraj/Road-Surface-Damage-Detection.git  
cd Road-Surface-Damage-Detection  

2. Install dependencies  
pip install -r requirements.txt  

3. Run the application  
streamlit run app.py  

4. Upload `potholes.zip` in the sidebar.

---

## 📊 Output

The system provides:
- Bounding box visualization on damaged regions
- Severity score for each image
- Risk classification (Low / Medium / High)
- Scatter plots and pie charts for analytics
- Downloadable CSV report

---

## 🧪 Dataset

The dataset is included in this repository as `potholes.zip`.  
It contains road images used for pothole and crack detection.

---

## 🎯 Use Cases

- Smart city road monitoring  
- Infrastructure quality analysis  
- Automated road inspection  
- Maintenance prioritization systems  

---

## 🔮 Future Enhancements

- Replace classical CV with deep learning (YOLO / CNN)
- Semantic segmentation of road damage
- GPS-based heatmap visualization
- Cloud deployment (AWS / GCP)

---

## 👨‍💻 Author

Raj Antala  
🎓 PGDM Student in AI and Data Science  
🏫 Adani Institute of Digital Technology Management (AIDTM)  
📍 Gandhinagar, India  
📧 antalaraj214@gmail.com  
🔗 www.linkedin.com/in/antalaraj

---

## ⭐ If you like this project, give it a star!
