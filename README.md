## 🎯 Objective

this project is to evaluate understanding of advanced supervised learning regression techniques, with a strong focus on regularization, model generalization, cross-validation strategies, and tree-based regression algorithms. Students will learn how to control overfitting, select optimal models, and compare linear vs non-linear regressors using real-world data.

---

## 📄 Problem Statement

You are hired as a **Junior Data Scientist** working on a real-estate analytics team. The company holds a **House Price dataset** of 3,800 properties and wants a **robust regression engine** that can predict a house's market price from its physical, locational and neighbourhood attributes.

Your manager asks you to build and compare multiple advanced regression approaches, control overfitting with regularization and cross-validation, evaluate tree-based and kernel-based models, and deliver a final comparison recommending which model the business should use.

The dataset contains:

- **Physical attributes** — area, bedrooms, bathrooms, property age.
- **Locational attributes** — location score, distance to city.
- **Neighbourhood indicators** — near school, near metro, crime rate index.
- **Target variable** — House Price (₹).

---



# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Robust_Regression_Engine.ipynb` | Main notebook — the complete, annotated regularization, cross-validation, tree, SVR and model-comparison pipeline |
| 📊 `Advanced_Regression_HousePrice_Dataset_3800 - Advanced_Regression_HousePrice_Dataset_3800.csv.csv` | Raw housing dataset (3,800 records) |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |
| 📄 `Part A :- Conceptual_Foundation.pdf` | Project documentation of part :- A (Theory) |


---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Regularization-Ridge%20%7C%20Lasso-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Cross--Validation-KFold%20%7C%20Stratified%20%7C%20LOOCV%20%7C%20TimeSeries-059669?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Tree%20Models-Decision%20Tree%20%7C%20Random%20Forest-16A34A?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SVR-Linear%20%7C%20RBF-7C3AED?style=for-the-badge"/>
<img src="https://img.shields.io/badge/GridSearchCV-Tuning-0EA5E9?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-Add%20Your%20Link-blue?style=for-the-badge&logo=googledrive&logoColor=white)](ADD_YOUR_DEMO_LINK)

📹 Add a link to your project walkthrough video here.

---

### 🧬 Dataset Structure — House Price Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `property_id` | Integer | Unique identifier for each property | Not used as a model feature |
| `sale_date` | Object (date) | Date the property was sold | Not used as a feature; used only to sort data for Time Series Split |
| `area_sqft` | Integer | House area in square feet | Model feature — strongest price driver |
| `bedrooms` | Integer | Number of bedrooms | Model feature |
| `bathrooms` | Integer | Number of bathrooms | Model feature |
| `location_score` | Float | Locational desirability score | Model feature — second strongest driver |
| `property_age` | Integer | Age of the property in years | Model feature — negative effect on price |
| `distance_city_km` | Float | Distance from city center (km) | Model feature |
| `near_school` | Binary Int | Whether a school is nearby | Model feature |
| `near_metro` | Binary Int | Whether a metro station is nearby | Model feature |
| `crime_rate_index` | Float | Crime rate of the neighbourhood | Model feature — negative effect on price |
| `house_price_inr` | Integer | House price (₹) | 🎯 **Target variable** |

---


## 🧠 Part B : Dataset Understanding & Preparation

### 📋 Dataset Overview

```python
df = pd.read_csv("Advanced_Regression_HousePrice_Dataset_3800.csv")
df.head(); df.tail(); df.describe(); df.info(); df.shape
```

**Output:** 3,800 rows × 12 columns · no missing values

💡 **Insight:** Prices average about ₹2.07 crore (std ≈ ₹87 lakh); area ranges from 500 to 3,776 sq ft and property age goes up to 80 years. `property_id` is only an identifier and `sale_date` is text, so neither can be used directly as a numeric feature. 📊

---

### 6️⃣ Identify features and target variable

```python
X = df.drop("house_price_inr", axis=1)
y = df["house_price_inr"]
```

💡 **Insight:** The target is `house_price_inr` and every other column is a candidate feature. The target is continuous, so this is a **regression** problem. 🎯

---

### 7️⃣ Perform a train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Output:** X_train (3040, 11) · X_test (760, 11)

💡 **Insight:** The 80/20 split gives 3,040 training and 760 testing rows — enough data on both sides for reliable evaluation, and `random_state=42` keeps every model on the same test set. 🔀

---

### 8️⃣ Apply basic preprocessing (scaling where required)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

💡 **Insight:** Only the 9 numeric predictors are kept. The scaler is **fit on training data only**, which prevents data leakage. Scaling matters for Ridge, Lasso and SVR; tree models do not need it. ⚖️

---


## 📈 Part C : Regularized Linear Models

### 9️⃣ Implement Ridge Regression (L2)

```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

**Output:** Training MSE 6.25 × 10¹² · Validation MSE 6.55 × 10¹²

💡 **Insight:** Training and validation errors are very close, so the model generalises well with no overfitting. MSE is in squared rupees, so RMSE (≈ ₹25.6 lakh) is easier to interpret. 📐

---

### 🔟 Implement Lasso Regression (L1)

```python
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
```

**Output:** Training MSE 6.25 × 10¹² · Validation MSE 6.54 × 10¹²

💡 **Insight:** Lasso gives almost the same errors as Ridge, and no feature is fully removed at α = 1 — all 9 predictors are informative. L1 and L2 behave alike here because the data has low noise and little multicollinearity. 📏

---

### 1️⃣1️⃣ Tune the regularization parameter (α) using cross-validation

```python
alphas = [0.01, 0.1, 1, 10, 100]
ridge_grid = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5, scoring="neg_mean_squared_error")
lasso_grid = GridSearchCV(Lasso(), {"alpha": alphas}, cv=5, scoring="neg_mean_squared_error")
```

**Output:** Best Ridge α = **1** · Best Lasso α = **100**

💡 **Insight:** Ridge prefers mild shrinkage while Lasso chose the strongest value in the grid. Lasso's best α sits at the grid edge, so a wider range could be tested to confirm it. Scores barely change across α — the model is not very sensitive to regularization strength. 🎛️

---

### 1️⃣2️⃣ Compare Ridge and Lasso (training error, validation error, coefficient behavior)

| Feature | Ridge Coefficient | Lasso Coefficient |
|---------|-------------------|-------------------|
| `area_sqft` | +6,937,545 | +6,944,831 |
| `bedrooms` | +297,366 | +291,194 |
| `bathrooms` | +285,327 | +285,382 |
| `location_score` | +3,676,865 | +3,678,911 |
| `property_age` | −651,302 | −651,391 |
| `distance_city_km` | −30,170 | −28,738 |
| `near_school` | +16,971 | +16,803 |
| `near_metro` | +58,204 | +58,197 |
| `crime_rate_index` | −135,473 | −135,309 |

💡 **Insight:** Both models agree closely — `area_sqft` and `location_score` are the strongest price drivers, followed by `property_age` (negative). Lasso slightly shrinks weak features like `distance_city_km` and `bedrooms` but zeroes none, so Ridge and Lasso are practically equivalent here. 🏆

---


## 📘 Part D : Cross-Validation Strategies

### 1️⃣3️⃣ Apply and compare CV techniques

```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring="neg_mean_squared_error")
```

| Technique | Average MSE | Insight |
|-----------|-------------|---------|
| 🟦 **K-Fold** | 6.316 × 10¹² | Fold MSEs range from 5.83 × 10¹² to 7.13 × 10¹² — moderate spread depending on which rows are validated |
| 🟨 **Stratified K-Fold** (target binned into 5 quantiles) | 6.300 × 10¹² | Keeps each fold's price distribution similar; very close to plain K-Fold |
| 🟩 **Leave-One-Out** | 6.298 × 10¹² | Least biased estimate, but one fit per row makes it expensive |
| 🟥 **Time Series Split** (sorted by `sale_date`) | 6.379 × 10¹² | Highest error — trains on the past, validates on the future |

---

### 1️⃣4️⃣ Analyze how performance metrics vary across CV strategies

💡 **Insight:** All four strategies land within about **1.3%** of each other (6.30 × 10¹² to 6.38 × 10¹²), so the performance estimate is **robust**. Use K-Fold for speed, and Time Series Split when temporal order matters. 🔁

---



## 🌳 Part E : Tree-Based Regression Models

### 1️⃣5️⃣ Implement Decision Tree Regression

```python
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
```

**Output:** Training R² 1.0 · Validation R² 0.858 · Validation MSE 1.14 × 10¹³

💡 **Insight:** Training MSE of 0 means the tree memorised the training data — classic **overfitting**. An unrestricted tree grows until every leaf is pure, so it needs complexity control. 🌱

---

### 1️⃣6️⃣ Control tree complexity using hyperparameters

```python
tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
```

**Output:** Training MSE 7.50 × 10¹² · Validation MSE 9.39 × 10¹²

💡 **Insight:** Limiting depth and leaf sizes cuts validation MSE from 1.14 × 10¹³ to 9.39 × 10¹² (about an **18% improvement**). Training error rises, which is expected — the model trades memorisation for generalisation. ✂️

---

### 1️⃣7️⃣ Implement Random Forest Regression

```python
forest = RandomForestRegressor(
    n_estimators=100, max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42
)
```

**Output:** Training R² 0.926 · Validation R² 0.914 · Validation MSE 6.90 × 10¹²

💡 **Insight:** Averaging 100 trees reduces variance, so the forest generalises far better than a single tree, with only a small train–validation gap. 🌲

---

### 1️⃣8️⃣ Compare single-tree vs ensemble performance

| Model | Training MSE | Validation MSE | Validation R² |
|-------|--------------|----------------|----------------|
| Decision Tree (pruned) | 7.50 × 10¹² | 9.39 × 10¹² | 0.883 |
| Random Forest | 5.51 × 10¹² | 6.90 × 10¹² | 0.914 |

💡 **Insight:** Random Forest beats the Decision Tree with about **26% lower** validation MSE and higher R². 🥇

---




## ⚙️ Part F : Support Vector Regression

### 1️⃣9️⃣ Implement Support Vector Regression (Linear and RBF kernels)

```python
svr_linear = SVR(kernel="linear")
svr_rbf    = SVR(kernel="rbf")
```

| Kernel | Training MSE | Validation MSE | Validation R² |
|--------|--------------|----------------|----------------|
| Linear | 7.50 × 10¹³ | 8.07 × 10¹³ | −0.0019 |
| RBF | 7.51 × 10¹³ | 8.08 × 10¹³ | −0.0029 |

💡 **Insight:** R² ≈ 0 means the model does no better than predicting the mean price — **underfitting**. The likely cause is the target scale: prices are in crores while the default `C` and `epsilon` are tiny relative to that. 📉

---

### 2️⃣0️⃣ Tune hyperparameters (C, γ, ε)

```python
parameters = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.1, 0.5, 1]}
svr_grid = GridSearchCV(SVR(kernel="rbf"), parameters, cv=5, scoring="neg_mean_squared_error")
```

**Output:** Best Parameters: `C = 10`, `gamma = 0.1`, `epsilon = 0.1` · Tuned Validation R² −0.0027

💡 **Insight:** The best values sit at the grid edge and R² is unchanged. Far larger `C` (10⁵ to 10⁷) or a scaled target is needed — tuning within this small grid brought no real gain. 🎛️

---

### 2️⃣1️⃣ Compare SVR performance with linear and tree-based models

| Model | MSE | R² Score |
|-------|-----|----------|
| Ridge | 6.545 × 10¹² | 0.9187 |
| Decision Tree | 9.386 × 10¹² | 0.8835 |
| Random Forest | 6.905 × 10¹² | 0.9143 |
| SVR | 8.075 × 10¹³ | −0.0027 |

💡 **Insight:** Ranking by R²: **Ridge > Random Forest > Decision Tree >> SVR**. A simple linear model matches or beats complex ones, so the data has a strong linear structure. 🚀

---


## 📊 Part G : Model Comparison & Evaluation

### 2️⃣2️⃣ Evaluate all models using regression metrics

```python
mse  = mean_squared_error(actual, prediction)
mae  = mean_absolute_error(actual, prediction)
rmse = np.sqrt(mse)
r2   = r2_score(actual, prediction)
```

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Lasso** | ₹19,59,385 | ₹25,58,145 | **0.9187** |
| Ridge | ₹19,59,199 | ₹25,58,402 | 0.9187 |
| Random Forest | ₹19,36,690 | ₹26,27,672 | 0.9143 |
| Decision Tree | ₹23,25,304 | ₹30,63,631 | 0.8835 |
| SVR | ₹69,92,913 | ₹89,86,120 | −0.0027 |

💡 **Insight:** Lasso and Ridge are the best — predictions are typically within about 12% of the average price. Random Forest has the lowest MAE but a slightly higher RMSE, so it makes fewer small errors but a few larger ones. 🎯

---

### 2️⃣3️⃣ Compare regularized linear, tree-based and SVR models

💡 **Insight:**
- **Regularized linear models** — best overall: accurate, simple, fast and easy to interpret through coefficients.
- **Tree-based models** — Random Forest is competitive; the single tree is weaker. A good option if non-linear patterns exist.
- **SVR** — worst here, mainly due to scale and tuning issues rather than the algorithm itself.

---



### 2️⃣4️⃣ Identify signs of overfitting or underfitting

| Model | Training R² | Validation R² | Fit |
|-------|--------------|----------------|-----|
| Ridge | 0.9162 | 0.9187 | ✅ Good Fit |
| Lasso | 0.9162 | 0.9187 | ✅ Good Fit |
| Decision Tree (pruned) | 0.8994 | 0.8835 | ✅ Good Fit |
| Random Forest | 0.9262 | 0.9143 | ✅ Good Fit |
| SVR | −0.0066 | −0.0027 | ⚠️ Underfitting |

💡 **Insight:** Four models are a good fit with train and validation R² within about 0.01 to 0.02. SVR underfits (R² ≈ 0 on both sets). The unpruned tree from Q15 was the only overfit model, and pruning fixed it. ⚖️

---

## 📂 Project Workflow

1. **Dataset Understanding** → Overview, identify features and target
2. **Train/Test Split** → 80/20 split (3,040 / 760 rows)
3. **Preprocessing** → StandardScaler fit on training data only
4. **Regularized Models** → Ridge (L2) and Lasso (L1), α tuned with GridSearchCV
5. **Cross-Validation** → K-Fold, Stratified K-Fold, Leave-One-Out, Time Series Split
6. **Tree-Based Models** → Decision Tree (unconstrained and pruned) and Random Forest
7. **Support Vector Regression** → Linear and RBF kernels, tuning of C, γ, ε
8. **Model Comparison** → MSE, MAE, RMSE, R² across all models
9. **Fit Diagnostics** → Overfitting and underfitting analysis

---

## 📈 Results & Insights

- ✅ **Five model families** built and compared — Ridge, Lasso, Decision Tree, Random Forest, SVR
- ✅ **Lasso / Ridge Regression** selected as best — **R² 0.919**, RMSE ≈ ₹25.6 lakh, MAE ≈ ₹19.6 lakh
- ✅ **Random Forest** is a strong backup — R² 0.914 and the lowest MAE (≈ ₹19.4 lakh)
- ✅ **Four CV strategies** agree within about 1.3%, confirming a robust performance estimate
- ✅ **Pruning** cut the Decision Tree's validation MSE by about 18% and removed its overfitting
- ✅ **SVR** underfits (R² ≈ 0) and needs target scaling or a much larger `C`
- ✅ **Top price drivers** — `area_sqft` and `location_score`; `property_age` and `crime_rate_index` lower prices

---

## 📌 Expected Outcomes

- Understand how to **plan and execute a complete supervised-learning regression workflow**
- Apply **Ridge and Lasso regularization** and tune α with cross-validation
- Compare **K-Fold, Stratified, Leave-One-Out and Time Series CV** and reason about when to use each
- Build **tree-based and SVR models** and control their complexity with hyperparameters
- Diagnose **overfitting and underfitting** using train and validation metrics
- Translate model results into a **practical business recommendation**

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/robust-regression-engine.git
cd robust-regression-engine

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter

# launch the notebook
jupyter notebook Robust_Regression_Engine.ipynb
```

---

## 🚀 Future Scope

- [ ] Widen the Lasso α grid (1000+) to confirm the optimal value
- [ ] Scale the target (or use a much larger `C`) so SVR can learn properly
- [ ] Add ElasticNet, Gradient Boosting and XGBoost as further baselines
- [ ] Engineer features from `sale_date` (year, month) for time-aware modelling
- [ ] Hyperparameter-tune Random Forest (depth, number of trees)
- [ ] Deploy the best model behind a simple prediction API

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.
