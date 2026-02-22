**End-to-End Energy CO2 Regression ML Project**

**Project Overview**
- End-to-end machine learning regression project in **Python**
- Predicts **country-level annual CO2 emissions**
- Built using a **global sustainable energy dataset**
- Implemented in **Jupyter Notebook** with **scikit-learn pipelines**
- Includes **model interpretability** using **SHAP** and **feature importance analysis**

**Objective**
- Predict target variable:
  - `Value_co2_emissions_kt_by_country` (CO2 emissions in kilotons)

**Input Features (Examples)**
- Access to electricity
- Access to clean cooking fuels
- Electricity generation from:
  - Fossil fuels
  - Nuclear
  - Renewables
- Renewable energy share
- Primary energy consumption
- Energy intensity
- GDP / GDP per capita / GDP growth
- Population density
- Land area
- Latitude / Longitude
- Country (`Entity`)
- Year

**Dataset**
- **Name:** Global Data on Sustainable Energy
- **Rows:** 3,649
- **Columns:** 21
- **Countries (Entity):** 176
- **Year Range:** 2000–2020

**End-to-End Workflow**

**1) Data Preparation**
- Load CSV dataset
- Inspect schema and summary stats
- Check missing values
- Remove rows with missing target values

**2) Train/Test Split**
- Create train/test sets
- Use **stratified sampling** (based on binned CO2 emissions)
- Preserve target distribution across splits

**3) Exploratory Data Analysis (EDA)**
- Histograms / distributions
- Correlation analysis
- Scatter plots
- Geographic feature inspection (latitude/longitude)

**4) Feature Engineering**
- Create derived energy-related features
- Examples:
  - Total electricity generation
  - Ratio/share-based features
  - Composition indicators (fossil vs low-carbon)

**5) Preprocessing Pipeline (scikit-learn)**
- **Numerical pipeline**
  - Median imputation
  - Standard scaling
- **Categorical pipeline**
  - Most-frequent imputation
  - One-hot encoding (`Entity`)
- Combine with `ColumnTransformer`

**6) Model Training**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**7) Model Evaluation**
- RMSE (Root Mean Squared Error)
- Cross-validation for model comparison

**8) Hyperparameter Tuning**
- `GridSearchCV`
- `RandomizedSearchCV`
- Focus on improving Random Forest performance

**9) Feature Importance + SHAP (Interpretability)**
- Added a dedicated **Feature Importance + SHAP** section in the notebook
- Imported and used:
  - `shap`
  - `RandomForestRegressor`
  - `permutation_importance` (scikit-learn)
  - `BaseSRegressor` from `causalml` (for additional interpretation workflow)
- Attempted built-in CausalML feature importance using `model_s.get_importance()` inside a `try/except` block
- Computed **permutation importance** on the trained Random Forest model using:
  - `permutation_importance(rf, X_test, y_test, random_state=42)`
- Created and plotted top feature importances (top 20)
- Applied **SHAP** with:
  - `explainer = shap.Explainer(rf, X_train)`
  - `shap_values = explainer(X_test[:200])`
  - `shap.summary_plot(shap_values, X_test[:200], show=True)`
- Used `try/except` handling for SHAP execution to make the notebook robust if SHAP is unavailable or fails in some environments
- This adds **explainability** by showing how features impact model predictions, beyond just reporting RMSE

**10) Final Testing**
- Evaluate final tuned model on held-out test set
- Estimate confidence interval for test RMSE

**11) Model Persistence**
- Save trained model with `joblib`
- Reload model for future predictions

**Results (Notebook Run)**
- **Final Test RMSE:** ~44,139
- **95% CI for Test RMSE:** ~[22,480, 58,234]

**Tech Stack**
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SHAP
- CausalML
- joblib

**Repository Structure**
- `notebooks/INSY674_End_to_End_Energy_CO2.ipynb` — main notebook
- `data/global-data-on-sustainable-energy.csv` — dataset
- 'notebooks/INSY674_Assignment2_VanshikaSapra_clean.ipynb' - SHAP / interpretability analysis notebook
- `README.md`
- `.gitignore`

**How to Run**
1. Clone the repository
2. Install dependencies
3. Open Jupyter Notebook
4. Run cells in order

**Install Dependencies**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `shap`
- `causalml`
- `joblib`

**Key Skills Demonstrated**
- End-to-end ML workflow
- Data cleaning & preprocessing
- Feature engineering
- Regression modeling
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Model interpretability using SHAP
- Pipeline-based reproducible ML
- Model saving/loading




