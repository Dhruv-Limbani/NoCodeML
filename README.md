# No-Code Machine Learning Model Building Platform

Welcome to the No-Code ML Model Building Platform! This platform enables users to perform data cleaning, transformation, visualization, and machine learning model building without writing any code.

## Features

### 1. Data Upload and Display
- Upload your dataset in CSV format.
- View and explore the data directly in the app.

!["data_upload"](front-end/data_upload.png)

### 2. Data Cleaning
- Handle missing values:
  - Drop rows with missing values.

    !["drop_na"](front-end/drop_na.png)

  - Impute missing values using mean, median, or mode.
    
    !["impute"](front-end/handle_missing_vals.png)

- Replace values in columns.

    !["replace"](front-end/replace.png)

- Change column data types.
    
    !["change_dtype"](front-end/change_dtype.png)

- Detect and remove outliers using:
  - Interquartile Range (IQR)
  - Z-Score

  !["outliers"](front-end/outliers.png)

- Download Cleaned Data as CSV:

  !["download_cleaned_data"](front-end/download_cleaned_data.png)


### 3. Data Analysis and Visualization

!["EDA"](front-end/EDA.png)

- **Unique Values**: Explore unique values for selected columns.

    !["unique_vals"](front-end/unique_vals.png)

- **EDA (Exploratory Data Analysis):**
  - Univariate Analysis: Visualize single-variable distributions.

    !["univariate"](front-end/univariate.png)

  - Bivariate Analysis: Analyze relationships between two variables.

    !["bivariate"](front-end/bivariate.png)

  - Multivariate Analysis: Correlation matrix and heatmaps.

    !["multivariate"](front-end/multivariate.png)

- **Feature Importance:**
  - Calculate feature importance using methods like Random Forest, Chi-Square, ANOVA F-value, Point Biserial Correlation, and Mutual Information.

    !["feat_impt_1"](front-end/feat_impt_1.png)

    !["feat_impt_2"](front-end/feat_impt_2.png)

### 4. Data Preparation for Modeling
- Drop irrelevant columns.

    !["drop_feat"](front-end/drop_feat.png)

- Split data into train and test sets.

    !["train_test_split"](front-end/train_test_split.png)

- Normalize numerical data and encode categorical data using:
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
  - Standard Scaling
  - Min-Max Scaling

    !["feat_norm_encode"](front-end/feat_norm_encode.png)

- Handle class imbalance (coming soon).
- Perform dimensionality reduction (coming soon).

    !["class_imb_pca"](front-end/class_imb_pca.png)

### 5. Machine Learning Model Building
- **Model Training:**
  - Choose between classification or regression tasks.
  - Supported algorithms include Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines, and more.

    !["model_train"](front-end/model_train.png)

- **Model Testing:**
  - Evaluate trained models on test data.
  - Download trained models as pickle files.

    !["model_test"](front-end/model_test.png)

### 6. Notes and Downloads
- Take notes directly in the app.
- Download cleaned data, transformed data, models, and notes.

## Getting Started
1. Clone the repository:

   ```bash
   git clone https://github.com/Dhruv-Limbani/NoCodeML.git
   ```
2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser and upload your data to get started.

---