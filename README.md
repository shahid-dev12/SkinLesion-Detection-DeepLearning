# Skin Lesion Detection using Deep Learning

This repository contains the implementation of my **MSCS Thesis: Comparative Analysis of Deep Learning Models for Early Skin Lesion Detection**.  
The work focuses on **skin cancer classification** (melanoma, basal cell carcinoma, squamous cell carcinoma, etc.) using dermoscopic images and deep learning.

---

## 📌 Objectives
- Perform a **comparative analysis** of multiple CNN architectures (ResNet50, AlexNet, InceptionV3, GoogLeNet).  
- Evaluate performance across **two benchmark datasets**: ISIC 2019 and HAM10000.  
- Identify the **best-performing model** for accurate and early detection of skin lesions.  
- Ensure **robustness** by applying 5-fold cross-validation and **statistical significance testing (paired t-tests, p < 0.05)**.  

---

## 🗂️ Datasets
The experiments were conducted on two public datasets:

1. **[ISIC 2019](https://challenge.isic-archive.com/)**  
   - Contains thousands of dermoscopic images for multiple skin lesion categories.  

2. **[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**  
   - Large collection of multi-source dermatoscopic images with 7 skin lesion classes.  

> ⚠️ Datasets are **not included** in this repo. Please download them directly from the official sources.

---

## 🏗️ Model Architectures
The following deep learning models were implemented and compared:
- **AlexNet**  
- **ResNet50**  
- **InceptionV3**  
- **GoogLeNet (Inception v1)**  

---

## ⚙️ Implementation Details
- **Language:** Python  
- **Platform:** Kaggle Notebook with GPU  
- **Frameworks/Libraries:** TensorFlow (2.x), Keras, NumPy, Pandas, Matplotlib, scikit-learn  
- **Batch Size:** 64  
- **Epochs:** 10  
- **Optimizer:** Adam (LR = 0.001)  
- **Loss Function:** Categorical Crossentropy  
- **Data Augmentation:** Rotation, Zoom, Flip, Shift  
- **Activation Function:** ReLU  
- **Regularization:** Dropout layers to reduce overfitting  

---

## 📊 Results
- **Best Model:** InceptionV3  
- **Accuracy:** 90.06% (ISIC 2019), 88.12% (HAM10000)  
- Statistically validated using **paired t-tests (p < 0.05)** to confirm performance differences were significant.  

---

## 🚀 How to Run
### Option 1: Run on Kaggle
- Upload the notebook(s) to [Kaggle Notebooks](https://www.kaggle.com/code).  
- Enable **GPU runtime**.  
- Place the dataset files in the correct paths and run the cells.  

### Option 2: Run Locally
```bash
git clone https://github.com/YOURUSERNAME/SkinLesion-Detection-DeepLearning.git
cd SkinLesion-Detection-DeepLearning
pip install -r requirements.txt
python train_model.py
