Laptop Price Prediction
Project Overview
This project focuses on predicting laptop prices in Indian Rupees (₹) based on various hardware and brand specifications. The dataset used contains detailed information about laptops, including their brand, series, weight, display size, pixel density, touchscreen capability, graphic processor, RAM capacity, RAM type, SSD capacity, operating system, processor brand, and processor series. The goal is to preprocess the dataset, perform exploratory data analysis (EDA), and build machine learning models to predict laptop prices accurately.
Dataset
The dataset (laptop_dataset.csv) contains information about laptops with various features. After preprocessing, the cleaned dataset is saved as processed_laptop_data.csv. Key preprocessing steps include:

Removing columns with more than 30% missing values (except for Graphics Memory, where NaN is replaced with 0 to indicate no dedicated GPU).
Dropping rows with missing target variable (Price (Rs)).
Imputing missing values in columns like Weight, Display Size, RAM Type, SSD Capacity, and Display Touchscreen using statistical methods or manual annotation based on reliable sources.
Feature engineering to create simplified columns (e.g., OS, Processor_Brand, Processor_Series, graphics_type).
Filtering out outdated configurations (e.g., laptops with SSD < 128GB).

Exploratory Data Analysis (EDA)
The EDA process involved analyzing key features to understand their impact on laptop prices. Key insights include:

Operating System: Windows 10 and 11 have a wide price range, while macOS laptops have higher average prices with less variation. Linux, Chrome OS, and DOS are typically found in lower-priced laptops.
Weight: Missing values were imputed with the median due to outliers. Weight has a moderate impact on price.
Display Size: Missing values were manually imputed for three entries. Display size was cleaned to extract numeric values.
RAM Capacity: Common configurations are 8GB and 16GB. Higher RAM correlates with higher prices.
Processor: Processor brands (Intel, AMD, Apple) and series were extracted. Rows with outdated configurations were removed.
RAM Type: Missing values were imputed based on average price per RAM type. DDR5 and LPDDR5X are associated with higher-priced laptops.
Graphics: A new feature, graphics_type, was created to categorize laptops as having no graphics, integrated graphics, or dedicated graphics.
Series: Missing values were imputed by extracting series names from the Model column.
Pixel Density: Missing values were manually imputed where possible. Higher pixel density is associated with premium brands.
SSD Capacity: Missing values were imputed using a smart function based on brand, RAM, and price bins. Laptops with SSD < 128GB were excluded.
Display Touchscreen: Missing values were imputed, with most laptops confirmed as non-touchscreen. Touchscreen laptops have a slightly higher average price.

Models
Two machine learning models were trained to predict the log-transformed Price (Rs):

Linear Regression: A baseline model using one-hot encoding for categorical features and standard scaling for numerical features.
Support Vector Regression (SVR): An SVR model with an RBF kernel, tuned using GridSearchCV for parameters C, epsilon, and gamma. RobustScaler was used for numerical features to handle outliers.

Model Performance

Linear Regression:
R² Score: [Insert R² score from output]
MAE: [Insert MAE from output]


SVR (Tuned):
R² Score: [Insert R² score from output]
MAE: [Insert MAE from output]



The SVR model generally outperforms Linear Regression due to its ability to capture non-linear relationships in the data.
Prediction Example
For a sample laptop with the following specifications:

Brand: HP
Series: Pavilion
Weight: 1.50 kg
Display Size: 14.0 inches
Pixel Density: 157 PPI
Display Touchscreen: No (0)
Graphic Processor: Intel UHD
RAM Capacity: 8 GB
RAM Type: DDR4
SSD Capacity: 512 GB
OS: Windows 11
Processor Brand: Intel
Processor Series: i3
Graphics Memory: 0 GB

Predicted prices:

Linear Regression: [Insert predicted price]
SVR: [Insert predicted price]

Files

laptop_dataset.csv: Original dataset.
processed_laptop_data.csv: Cleaned and preprocessed dataset.
laptop_price_prediction.ipynb: Jupyter notebook containing the code for data preprocessing, EDA, and model training.
README.md: This file.

Requirements
To run the code, install the required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Clone the repository:git clone https://github.com/your-username/laptop-price-prediction.git


Install the required dependencies (see above).
Open the laptop_price_prediction.ipynb notebook in Jupyter Notebook or JupyterLab.
Run the notebook cells to preprocess the data, perform EDA, and train the models.
Use the trained models to make predictions on new data (as shown in the prediction example).

Future Improvements

Incorporate additional features like battery life or build material if available.
Experiment with advanced models like Random Forest, Gradient Boosting, or Neural Networks.
Perform feature selection to reduce dimensionality and improve model performance.
Collect more data to improve model robustness, especially for rare configurations.

Author

Gaurav Pandey NadeemInspired by observations as a laptop salesperson at Vijay Sales, India.

License
This project is licensed under the MIT License.
