# 🧠 Medical Image Classification with Explainable AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Model](https://img.shields.io/badge/Model-DenseNet121-green)
![Dataset](https://img.shields.io/badge/Dataset-MURA-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview
This project uses deep learning to classify **wrist X-ray images** as **abnormal** or **normal**.  
It also applies **Grad-CAM** to generate heatmaps that explain which regions of the image influenced the model’s decision.

---

## 🖼️ Example Grad-CAM Output

<p align="center">
  <img src="AI/gradcam_example.png" width="400"/>
</p>

> Grad-CAM highlights areas the model focuses on when making predictions.

---

## 🧠 Model
- **Architecture:** DenseNet121 (transfer learning)
- **Framework:** PyTorch
- **Task:** Binary classification (abnormal vs normal)
- **Explainability:** Grad-CAM

---

## 📊 Dataset
- **Source:** Stanford MURA dataset
- **Subset:** Wrist X-rays

| Split | Size |
|------|------|
| Train | 24,597 |
| Validation | 3,646 |
| Test | 7,289 |

---

## 📈 Results

| Metric | Abnormal | Normal |
|--------|---------|--------|
| Precision | 0.74 | 0.77 |
| Recall | 0.63 | 0.85 |
| F1 Score | 0.68 | 0.81 |

- **Test Accuracy:** ~76%  
- **Validation Accuracy:** ~76.8%  

---

## 🔍 Confusion Matrix

<p align="center">
  <img src="AI/confusion_matrix.png" width="400"/>
</p>

---

## ⚙️ Features
- Transfer learning with pretrained DenseNet121  
- Class-weighted loss for imbalance  
- Validation-based model checkpointing  
- Grad-CAM visualization for interpretability  

---

## ⚠️ Limitations
- Lower recall for abnormal cases  
- Dataset imbalance (more normal images)  
- Limited to wrist X-rays  

---

## 🚀 Future Improvements
- Improve abnormal detection (recall)
- Increase image resolution (224×224+)
- Try EfficientNet or other architectures
- Expand to more body parts

---

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt
python medical_densenet121_val_split.py
