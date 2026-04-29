# 🌬️ ReneWind — Wind Turbine Failure Prediction using Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

ReneWind is a company working on improving the machinery and processes involved in wind energy production using machine learning. This project builds and evaluates multiple **Artificial Neural Network (ANN)** models to predict wind turbine generator failures from sensor data, enabling proactive maintenance before costly breakdowns occur.

The nature of predictions:
- **True Positive (TP)** — Failure correctly detected → Repair cost (medium)
- **False Negative (FN)** — Failure missed → Full replacement cost (highest)
- **False Positive (FP)** — False alarm → Inspection cost only (lowest)

Since missing a real failure is far more costly than a false alarm, **Recall** is used as the primary evaluation metric.

---

## 🎯 Objective

> Build a binary classification model that predicts wind turbine generator failures so that generators can be repaired before failing, reducing overall maintenance costs.

- **Target Variable:** `Target` (1 = Failure, 0 = No Failure)
- **Task Type:** Binary Classification
- **Primary Metric:** Recall

---

## 📂 Dataset

| Property | Details |
|---|---|
| Training Set | 20,000 observations |
| Test Set | 5,000 observations |
| Features | 40 numerical sensor variables (V1–V40) |
| Target | Binary (1 = Failure, 0 = No Failure) |
| Feature Type | All continuous (float64) |
| Missing Values | V1 and V2 only |
| Class Distribution | ~94.5% No Failure, ~5.5% Failure |

> ⚠️ Data is ciphered — the exact nature of each sensor reading is confidential.

---

## 🔍 Exploratory Data Analysis

- Strong class imbalance — only ~5.5% of observations are failures
- All 40 features are numerical — no encoding required
- Missing values found only in V1 and V2 (in both train and test sets)
- Several features exhibit skewed distributions and outliers
- Moderate correlations observed between some feature pairs — no extreme multicollinearity
- Neural networks are well-suited for this type of high-dimensional sensor data

---

## ⚙️ Data Preprocessing

| Step | Treatment |
|---|---|
| Duplicate Check | No duplicates found |
| Missing Value Imputation | Median imputation (fit on train only, applied to val & test) |
| Outlier Treatment | No removal — neural networks are robust; removal risks losing failure signal |
| Feature Engineering | None required — all features already numerical |
| Train/Val Split | 80/20 stratified split (random_state=1) |
| Class Imbalance | Class weights computed and applied in Models 3 & 6 |

---

## 🧠 Model Building

All models are **Artificial Neural Networks (ANNs)** built using TensorFlow/Keras Sequential API.

**Common Settings:**
- Epochs: 50
- Batch Size: 32
- Loss: Binary Crossentropy
- Output: 1 neuron, Sigmoid activation
- Metric: Recall

| Model | Architecture | Optimiser | Dropout | Class Weight |
|---|---|---|---|---|
| Model 0 | 1 hidden (7) | SGD | No | No |
| Model 1 | 2 hidden (16→8) | SGD | No | No |
| Model 2 | 3 hidden (16→8→4) | SGD | 50% | No |
| Model 3 | 3 hidden (16→8→4) | SGD | 50% | Yes |
| **Model 4 ★** | **2 hidden (16→8)** | **Adam** | **No** | **No** |
| Model 5 | 3 hidden (32→16→8) | Adam | 50% | No |
| Model 6 | 3 hidden (32→16→8) | Adam | 50% | Yes |

---

## 📊 Model Performance Summary

**Training Set:**

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Model 0 | 0.9865 | 0.8916 | 0.9764 | 0.9294 |
| Model 1 | 0.9921 | 0.9354 | 0.9879 | 0.9600 |
| Model 2 | 0.9868 | 0.8849 | 0.9876 | 0.9294 |
| Model 3 | 0.9666 | 0.9341 | 0.8183 | 0.8656 |
| **Model 4** | **0.9942** | **0.9497** | **0.9946** | **0.9710** |
| Model 5 | 0.9913 | 0.9239 | 0.9929 | 0.9554 |
| Model 6 | 0.9913 | 0.9239 | 0.9929 | 0.9554 |

**Validation Set:**

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Model 0 | 0.9868 | 0.8976 | 0.9725 | 0.9314 |
| Model 1 | 0.9885 | 0.9176 | 0.9698 | 0.9419 |
| Model 2 | 0.9873 | 0.8936 | 0.9827 | 0.9331 |
| Model 3 | 0.9633 | 0.9191 | 0.8064 | 0.8523 |
| **Model 4** | **0.9913** | **0.9318** | **0.9835** | **0.9559** |
| Model 5 | 0.9893 | 0.9095 | 0.9866 | 0.9443 |
| Model 6 | 0.9893 | 0.9095 | 0.9866 | 0.9443 |

---

## 🏆 Best Model — Model 4

**Architecture:**
```
Input (40 features)
       ↓
Dense (16 neurons, ReLU)
       ↓
Dense (8 neurons, ReLU)
       ↓
Dense (1 neuron, Sigmoid)
```

**Why Model 4?**
- Highest validation Recall of **93.18%**
- Minimal overfitting — train-validation gap of only **1.8%**
- Adam optimiser converges faster and more effectively than SGD
- Best balance between Recall, Precision, and generalisation

**Test Set Performance:**

| Metric | Score |
|---|---|
| Accuracy | 99.13% |
| Recall | 93.17% |
| Precision | 98.35% |
| F1 Score | 95.59% |

> The model successfully identifies **93 out of every 100 actual turbine failures**, directly reducing costly unplanned generator replacements.

---

## 💡 Actionable Insights & Recommendations

- **Deploy Model 4** for real-time turbine monitoring to catch failures before breakdown
- **Prioritise early inspection** when the model flags a potential failure — repair cost is far lower than replacement
- **Investigate sensors V1 & V2** — missing values suggest possible calibration issues; improving sensor reliability will improve model accuracy over time
- **Retrain periodically** as new sensor data accumulates to adapt to changing turbine degradation patterns and seasonal effects

---

## 🛠️ Tech Stack

| Library | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.18.0 |
| Scikit-learn | 1.3.2 |
| Pandas | 2.2.2 |
| NumPy | 1.26.4 |
| Matplotlib | 3.8.3 |
| Seaborn | 0.13.2 |

---

## 📁 Project Structure

```
ReneWind/
│
├── data/
│   ├── Train.csv               # Training dataset (20,000 rows)
│   └── Test.csv                # Test dataset (5,000 rows)
│
├── notebook/
│   └── ReneWind_Notebook.ipynb # Main project notebook
│
├── presentation/
│   └── ReneWind_Presentation.pptx  # Project presentation slides
│
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/renewind-failure-prediction.git
cd renewind-failure-prediction
```

2. **Install dependencies**
```bash
pip install tensorflow==2.18.0 scikit-learn==1.3.2 pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.3 seaborn==0.13.2
```

3. **Add the data files**
- Place `Train.csv` and `Test.csv` inside the `data/` folder

4. **Run the notebook**
```bash
jupyter notebook notebook/ReneWind_Notebook.ipynb
```

5. **Run all cells sequentially** from top to bottom

---

## 📜 License

This project is for educational purposes as part of a Machine Learning course.

---

## 🙌 Acknowledgements

- **ReneWind** for providing the sensor dataset
- **Great Learning** for project guidance and structure
- **TensorFlow / Keras** for the deep learning framework
