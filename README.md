# SignSmart: Traffic Signal Recognition

SignSmart is an AI-powered Traffic Signal Recognition system built using deep learning. It classifies traffic signs into 43 categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## ✨ Features
- **Custom CNN Architecture**: A robust multi-layer CNN with Batch Normalization and Dropout.
- **Pre-trained MobileNetV2**: Leverage transfer learning for high-performance classification.
- **Data Augmentation**: Advanced augmentation (rotation, zoom, shift) to improve model generalization.
- **Comprehensive Evaluation**: Automated plotting of training history and confusion matrices.

## 🛠 Prerequisites
You will need **Python 3.8+** and the following libraries:
- TensorFlow / Keras
- OpenCV
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

Install them using:
```bash
pip install -r requirements.txt
```

## 📂 Dataset Setup
1. Download the GTSRB dataset from [Kaggle: GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Extract the files and place them into the `dataset/` folder.
3. Ensure the folder structure is as follows:
   ```text
   SignSmart/
   ├── dataset/
   │   ├── Train/
   │   │   ├── 0/
   │   │   ├── 1/
   │   │   └── ... (folders 0-42)
   │   ├── Test/
   │   ├── Train.csv
   │   └── Test.csv
   ```

## 🚀 How to Run
### 1. Training the Custom CNN
This is the default model.
```bash
python main.py --model-type custom --epochs 20 --batch-size 32
```

### 2. Training with MobileNetV2 (Transfer Learning)
This will use a pre-trained MobileNetV2 with ImageNet weights.
```bash
python main.py --model-type pretrained --epochs 10 --batch-size 16
```

## 📊 Evaluation
Trained models are saved in the `models/` folder. Training curves (accuracy/loss) and the final confusion matrix are automatically saved in the `plots/` directory after execution.

## 🏆 Bonus Tasks Completed
- [x] Data Augmentation implemented in `trainer.py`.
- [x] Model comparison (Custom CNN vs. MobileNetV2) available via `--model-type`.
