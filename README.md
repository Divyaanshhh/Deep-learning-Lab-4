# ANN Lab 4 - Feedforward Neural Network for Student Academic Performance Classification

A practical implementation of a **Multi-Layer Feedforward Neural Network** using **TensorFlow/Keras** to predict student academic performance grades based on key academic and behavioral features.

---

## 📌 Objective
To build and train a multi-layer neural network that classifies students into performance categories (**Grade A, Grade B, Grade C**) using academic metrics, and to understand how multiclass classification works with the **Softmax activation function** and **Sparse Categorical Crossentropy** loss.

---

## 🧠 What is a Multiclass Classification Neural Network?

Multiclass classification is the problem of classifying instances into one of three or more classes. Unlike binary classification (which has only two outcomes), multiclass problems require the model to choose among multiple categories.

### Architecture Overview
```
Input Layer (4 neurons)  →  Hidden Layer 1 (16 neurons, ReLU)
                                    ↓
                         Hidden Layer 2 (8 neurons, ReLU)
                                    ↓
                         Output Layer (3 neurons, Softmax)
```

### Key Components
| Component | Description |
|:---|:---|
| **Input Layer** | Receives normalized student feature values |
| **Hidden Layers** | Learn non-linear relationships between features and grades |
| **Softmax Activation** | Converts raw outputs into probability scores that sum to 1 across all 3 classes |
| **Sparse Categorical Crossentropy** | Loss function for integer-labeled multiclass problems |
| **Argmax Prediction** | Selects the class with the highest probability as the final prediction |

### Why Softmax for Multiclass?
The **Softmax** function is essential for multiclass classification because it:
- Squashes output values into a probability distribution (0 to 1)
- Ensures all class probabilities sum to exactly 1.0
- Allows the model to express confidence across multiple classes simultaneously

---

## 🛠️ Technologies Used
- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn (train_test_split, StandardScaler, metrics)
- TensorFlow 2.x / Keras

---

## 📁 Project Structure
```
ANN-Lab-4/
│── student_performance.ipynb       # Student Grade Prediction (Jupyter/Colab)
│── README.md
└── requirements.txt
```

---

## 🚀 Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ANN-Lab-4.git
```

### 2. Open the Project
```bash
cd ANN-Lab-4
```

### 3. Create a Virtual Environment
**Windows**
```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment
**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```
**Command Prompt**
```cmd
.venv\Scripts\activate
```

### 5. Install Dependencies
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### 6. Run the Notebook
Open `student_performance.ipynb` in **Jupyter Notebook**, **JupyterLab**, or **Google Colab** and run all cells.

---

## ⚙️ Program Description

### 🎯 Problem Statement
Educational institutions often struggle to identify at-risk students before final examinations. By analyzing early-semester metrics, a neural network can predict which students are likely to achieve top grades, average grades, or need academic intervention — enabling counselors to provide targeted support.

### 📊 Feature Engineering
We extract four key academic indicators for each student:

| Feature | Description | Typical Range |
|:---|:---|:---:|
| **Attendance (%)** | Percentage of classes attended | 60 – 100 |
| **Assignment Score** | Average score across submitted assignments | 0 – 100 |
| **Midterm Score** | Score obtained in midterm examinations | 0 – 100 |
| **Study Hours/Week** | Self-reported hours spent studying per week | 0 – 40 |

These features are chosen because they are strong predictors of academic success:
- **Attendance**: Regular class attendance correlates strongly with concept retention and exam performance.
- **Assignment Score**: Reflects consistent effort, understanding of coursework, and time management.
- **Midterm Score**: An early indicator of subject mastery and identifies knowledge gaps before finals.
- **Study Hours**: Quantifies out-of-class effort and self-discipline, both critical for deep learning.

### 📊 Training Dataset
The dataset contains **150 student records** divided equally among three performance categories:

| Grade Category | Description | Count |
|:---:|:---|:---:|
| **Grade C** | Needs Improvement (low attendance, low scores, minimal study) | 50 |
| **Grade B** | Satisfactory (moderate metrics across all features) | 50 |
| **Grade A** | Excellent (high attendance, high scores, extensive study) | 50 |

### 🔧 Preprocessing: Standardization
Since features operate on different scales (Attendance: 60-100, Study Hours: 0-40), we apply **StandardScaler** to normalize each feature to zero mean and unit variance. This prevents features with larger magnitudes from dominating the learning process.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 🔄 How the Neural Network Learns
1. **Forward Propagation**: Input features pass through two hidden layers with ReLU activation.
2. **Softmax Output**: The final layer outputs 3 probability values (one per grade class) using Softmax.
3. **Loss Calculation**: Sparse Categorical Crossentropy measures the difference between predicted probabilities and the true integer label.
4. **Backpropagation**: The Adam optimizer updates weights to minimize loss over 50 epochs.
5. **Validation**: 20% of training data is held out to monitor overfitting during training.

### 🏗️ Model Architecture
```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),   # Hidden Layer 1: 16 neurons
    Dense(8, activation='relu'),                      # Hidden Layer 2: 8 neurons
    Dense(3, activation='softmax')                    # Output Layer: 3 neurons (Softmax)
])
```

| Layer | Neurons | Activation | Purpose |
|:---|:---:|:---:|:---|
| Input | 4 | — | Receives standardized student metrics |
| Hidden 1 | 16 | ReLU | Learns basic patterns (e.g., high attendance + high study = better grade) |
| Hidden 2 | 8 | ReLU | Combines patterns into higher-level grade predictions |
| Output | 3 | Softmax | Outputs probability distribution across Grade C, B, and A |

### ✅ Expected Output
```
Test Accuracy: 0.9667

Classification Report

              precision    recall  f1-score   support

     Grade C       1.00      1.00      1.00        10
     Grade B       0.91      1.00      0.95        10
     Grade A       1.00      0.91      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```
> *Note: Exact accuracy may vary slightly due to random initialization, but should consistently exceed 90%.*

### 📈 Visualizations
The notebook includes:
- **Confusion Matrix**: Shows correct vs. misclassified predictions per grade category
- **Accuracy Plot**: Tracks training and validation accuracy across epochs
- **Loss Plot**: Tracks training and validation loss to detect overfitting

---

## 🌍 Real-World Applications of Multiclass Neural Networks

| Domain | Application |
|---|---|
| **Education** | Student grade prediction, dropout risk assessment, personalized learning paths |
| **Healthcare** | Disease type classification, tumor grading, symptom-based diagnosis |
| **Finance** | Credit rating classification (Poor/Fair/Good/Excellent), risk segmentation |
| **Retail** | Customer segmentation (Budget/Regular/Premium), product categorization |
| **Agriculture** | Crop recommendation based on soil type, weather-based yield classification |
| **Manufacturing** | Product quality grading (Defective/Acceptable/Premium), fault classification |
| **Human Resources** | Employee performance rating, candidate ranking systems |

---

## ⚠️ Limitations
- **Synthetic Dataset**: This demo uses synthetically generated data. Real-world deployment requires historical institutional data.
- **Small Feature Set**: Only 4 features are used. Real academic prediction models may include socioeconomic factors, prior GPA, extracurriculars, etc.
- **Class Imbalance**: The dataset is perfectly balanced (50 each). Real student populations may be skewed toward average performers.
- **No Temporal Data**: The model does not account for improvement trends over time.

> These limitations motivate advanced techniques like **LSTM networks** (for time-series academic tracking), **larger feature engineering**, and **imbalanced dataset handling** (SMOTE, class weighting).

---

## 📚 Concepts Covered
- Multiclass Classification
- Feedforward Neural Network (FNN)
- Softmax Activation Function
- Sparse Categorical Crossentropy Loss
- ReLU Activation
- Hidden Layers
- StandardScaler / Feature Standardization
- Train-Test Split with Stratification
- Confusion Matrix & Classification Report
- Model Validation & Overfitting Detection
- TensorFlow / Keras

---

## 🔖 References
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/

---

## 👤 Author
**Divyansh**
BSc Data Science and AI, Christ University
Deep Learning Laboratory — BDA404-5N
