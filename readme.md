# 💻 Laptop Price Prediction

## 📝 Project Overview

This project focuses on predicting laptop prices in Indian Rupees (₹) based on various hardware and brand specifications. The dataset contains detailed information about laptops, including:

* Brand
* Series
* Weight
* Display size
* Pixel density
* Touchscreen capability
* Graphic processor
* Graphic Memory
* RAM capacity
* RAM Type
* SSD Capacity
* Operating system
* Processor brand & series

The goal is to preprocess the dataset, perform exploratory data analysis (EDA), and build machine learning models to predict laptop prices accurately.

<img width="643" height="418" alt="image" src="https://github.com/user-attachments/assets/9320e335-77cb-48fe-8021-a5c422404d73" />

---

## 📂 Dataset

* **Original File**: `laptop_dataset.csv`
* **Processed File**: `processed_laptop_data.csv`

### Key Preprocessing Steps:

* Removed columns with >30% missing values (except `Graphics Memory`, where NaN = 0).
* Dropped rows missing the target variable (`Price (Rs)`).
* Imputed missing values in `Weight`, `Display Size`, `RAM Type`, `SSD Capacity`, `Display Touchscreen`.
* Feature engineering for simplified columns like `OS`, `Processor_Brand`, `Processor_Series`, `graphics_type`.
* Filtered out outdated configurations (e.g., SSD < 128GB).

---

## 📊 Exploratory Data Analysis (EDA)

### 🔍 Key Insights:

* **Operating System**:

  * Windows 10/11: Wide price range.
  * macOS: Higher average price, less variation.
  * Linux/Chrome OS/DOS: Budget laptops.

* **Weight**:

  * Median imputation used due to outliers.
  * Moderate impact on price.

* **Display Size**:

  * Cleaned & manually imputed where missing.

* **RAM Capacity**:

  * 8GB and 16GB most common.
  * Higher RAM = Higher price.

* **Processor**:

  * Extracted `Processor_Brand` and `Processor_Series`.
  * Removed outdated processors.

* **RAM Type**:

  * DDR5 & LPDDR5X linked to premium laptops.

* **Graphics**:

  * Engineered `graphics_type`: no, integrated, or dedicated GPU.

* **Pixel Density**:

  * Higher PPI → Premium pricing.

* **SSD Capacity**:

  * Imputed using smart logic.
  * Excluded SSD < 128GB.

* **Display Touchscreen**:

  * Most laptops non-touchscreen.
  * Touchscreen = Slightly higher price.

---

## 🤖 Models

We experimented with three machine learning models to predict **log-transformed** laptop prices:

### 1. **Linear Regression**

- One-hot encoding for categorical features.
- `StandardScaler` for numerical features.

### 2. **Support Vector Regression (SVR)**

- Used RBF kernel.
- Tuned using `GridSearchCV` with cross-validation.
- `RobustScaler` used for outlier resistance.

### 3. **Artificial Neural Network (ANN)**

- Built using `Keras` with dense layers and dropout.
- Hyperparameter tuning via 5-fold cross-validation.
- Final model trained with early stopping and tuned parameters.

**ANN Architecture Highlights**:

- Input layer with `ReLU` activation and dropout.
- Multiple hidden layers (tunable).
- Output layer with linear activation (for regression).
- Optimized with `Adam` and trained on log prices.
- Early stopping to prevent overfitting.

---

## 📈 Model Performance

| Model             | R² Score | MAE (₹) | RMSE (₹) |
|-------------------|----------|---------|----------|
| Linear Regression | `0.78`   | `8800`  | `11000`  |
| SVR (Tuned)       | `0.86`   | `7200`  | `9500`   |
| ANN (Tuned)       | `0.88`   | `6700`  | `9100`   |

> ✅ SVR outperforms Linear Regression and ANN by capturing non-linear relationships in the data.

---

## 🧪 Prediction Example

For the following laptop specs:

* Brand: **HP**
* Series: **Pavilion**
* Weight: **1.50 kg**
* Display Size: **14.0 inches**
* Pixel Density: **157 PPI**
* Display Touchscreen: **No (0)**
* Graphic Processor: **Intel UHD**
* RAM Capacity: **8 GB**
* RAM Type: **DDR4**
* SSD Capacity: **512 GB**
* OS: **Windows 11**
* Processor Brand: **Intel**
* Processor Series: **i3**
* Graphics Memory: **0 GB**

**Predicted Prices**:

* **Linear Regression**: ₹52,000
* **SVR**: ₹55,500

---

## 📁 Files

* `laptop_dataset.csv`: Original dataset
* `processed_laptop_data.csv`: Cleaned dataset
* `laptop_price_prediction.ipynb`: Jupyter notebook for preprocessing, EDA, and modeling
* `README.md`: Project documentation

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/laptop-price-prediction.git
```

2. Install required libraries (see above).

3. Open the notebook:

```bash
jupyter notebook nb_eda.ipynb
jupyter notebook nb_models.ipynb
```

4. Run all cells to preprocess, explore, train, and predict.

---

## 🔮 Future Improvements

* Add features like battery life, build material, or screen refresh rate.
* Test advanced models: Random Forest, XGBoost, or Neural Networks.
* Apply feature selection to improve performance.
* Expand dataset to include more laptop models and rare configurations.

---
