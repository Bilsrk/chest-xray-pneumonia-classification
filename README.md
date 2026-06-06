# chest-xray-pneumonia-classification
Deep learning classification of chest X-rays using ResNet50 and ResNet101
Get the resnet101 models from the link below,I was only able to get 2 runs in before my T4 gpu time is up so I customized it to save both runs seperately so I dont loose my progress if the T4 times gets over in the middle of a run.
Will be getting more runs in soon and uploading the agregate model soon!
For now here is the drive link for accessing both run pretrained models:- [text](https://drive.google.com/drive/folders/1OtUXwyfmzWhBRVhIGLNXE6iGOCi_a5MV?usp=drive_link)
The pre-trained weight model for resnet50 is in the project files itself.

# Chest X-Ray Pneumonia Classification

Automated classification of chest X-ray images into **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia** using deep learning and transfer learning with PyTorch.

---

## Overview
Pneumonia kills over 2.5 million people annually, particularly in regions where specialist radiologists are scarce. This project develops and evaluates deep learning models to automatically classify chest X-rays, supporting faster and more consistent diagnosis in resource-limited healthcare settings.

---

## Dataset
[Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Class | Count | Proportion |
|---|---|---|
| Normal | 1,583 | 27.0% |
| Bacterial Pneumonia | 2,780 | 47.5% |
| Viral Pneumonia | 1,493 | 25.5% |
| **Total** | **5,856** | **100%** |

---

## Key Features
- **Transfer Learning** from ImageNet pretrained ResNet50/101 weights
- **Differential Learning Rates** — gentle fine-tuning of pretrained layers
- **WeightedRandomSampler** — handles class imbalance during training
- **Early Stopping** with patience=5 to prevent overfitting
- **Cosine Annealing** learning rate scheduler for smooth convergence
- **Label Smoothing** to prevent overconfident predictions
- **Test Time Augmentation (TTA)** — 5-pass averaging at inference
- **Grad-CAM Visualizations** — heatmaps showing model focus regions
- **5-run evaluation protocol** with average and standard deviation reported

---

## Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize which regions of the X-ray the model focuses on. Results confirm the model attends to clinically relevant lung regions rather than background artifacts — a key requirement for clinical deployment.

---

## Tech Stack
- **Framework:** PyTorch 2.7.1 + CUDA
- **Models:** torchvision ResNet50, ResNet101
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib, OpenCV (Grad-CAM)
- **Data:** NumPy, Pandas, Pillow
- **Environment:** Anaconda (Python 3.11), Google Colab (T4 GPU)
- **Version Control:** Git, GitHub

---

## Setup and Usage

### Requirements
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn Pillow opencv-python
```

### Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract to your project folder
3. Update `DATASET_PATH` in the notebook to your local path

### Running
1. Open `Resnet_pneumonia.ipynb` for ResNet50
2. Open `resnet101_final.ipynb` for ResNet101
3. Run cells top to bottom

---

## Project Structure
chest-xray-pneumonia-classification/
├── Resnet_pneumonia.ipynb       ← ResNet50 full pipeline (5 runs, Anaconda)
├── resnet101_final.ipynb        ← ResNet101 full pipeline (Colab T4 GPU)
├── resnet_perrun_epochs.txt     ← Training logs per run
├── README.md                    ← Project documentation
└── .gitignore                   ← Excludes dataset and model weights

---

## Future Work
- Ensemble methods combining ResNet50 + ResNet101
- Training on CheXNet 100,000+ X-ray dataset
- Extending to tuberculosis and lung cancer detection
- DICOM integration for hospital radiology systems
- Mobile deployment via ONNX for point-of-care use
- Clinical validation toward regulatory approval

---

## Author
**Bilsrk** — Final Year CS Student
- GitHub: [@Bilsrk](https://github.com/Bilsrk)
- LinkedIn: [Your LinkedIn URL]

---

## References
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Selvaraju et al. (2017). Grad-CAM. ICCV.
- Mooney (2018). Chest X-Ray Images (Pneumonia). Kaggle.
- Rajpurkar et al. (2017). CheXNet. arXiv:1711.05225.