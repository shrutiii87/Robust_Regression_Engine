import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Robust Regression Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

[data-testid="stSidebar"] {
    background-color: #171a21;
    border-right: 1px solid #30343d;
}

[data-testid="stSidebar"] h2 {
    color: #ffffff;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 18px;
    color: #aeb4c0;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
    color: #ffffff;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: #171a21;
    border: 1px solid #30343d;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}

.metric-title {
    color: #aeb4c0;
    font-size: 15px;
}

.metric-value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.prediction-box {
    background: #171a21;
    border: 1px solid #3d4350;
    border-radius: 14px;
    padding: 22px;
    margin-top: 20px;
}

.prediction-title {
    color: #aeb4c0;
    font-size: 16px;
}

.prediction-price {
    color: #ffffff;
    font-size: 34px;
    font-weight: 700;
    margin-top: 8px;
}

.info-box {
    background: #171a21;
    border: 1px solid #30343d;
    border-radius: 14px;
    padding: 18px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

file_name = (
    "Advanced_Regression_HousePrice_Dataset_3800 "
    "- Advanced_Regression_HousePrice_Dataset_3800.csv (1).csv"
)

if os.path.exists(file_name):
    csv_file = file_name
else:
    csv_files = glob.glob(
        "Advanced_Regression_HousePrice_Dataset_3800*.csv"
    )

    if len(csv_files) == 0:
        st.error("Dataset CSV file not found.")
        st.stop()

    csv_file = csv_files[0]


df = pd.read_csv(csv_file)


# =========================================================
# FEATURES FROM YOUR PROJECT
# =========================================================

features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "location_score",
    "property_age",
    "distance_city_km",
    "near_school",
    "near_metro",
    "crime_rate_index"
]

target = "house_price_inr"


X = df[features]
y = df[target]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# RIDGE - SAME AS PROJECT
# =========================================================

ridge_grid = GridSearchCV(
    Ridge(),
    {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_grid.fit(
    X_train_scaled,
    y_train
)

best_ridge = ridge_grid.best_estimator_


# =========================================================
# LASSO - SAME AS PROJECT
# =========================================================

lasso_grid = GridSearchCV(
    Lasso(),
    {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="neg_mean_squared_error"
)

lasso_grid.fit(
    X_train_scaled,
    y_train
)

best_lasso = lasso_grid.best_estimator_


# =========================================================
# DECISION TREE - SAME AS PROJECT
# =========================================================

tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

tree.fit(
    X_train,
    y_train
)


# =========================================================
# RANDOM FOREST - SAME AS PROJECT
# =========================================================

forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

forest.fit(
    X_train,
    y_train
)


# =========================================================
# SVR - SAME TUNING AS PROJECT
# =========================================================

svr_grid = GridSearchCV(
    SVR(kernel="rbf"),
    {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1],
        "epsilon": [0.1, 0.5, 1]
    },
    cv=5,
    scoring="neg_mean_squared_error"
)

svr_grid.fit(
    X_train_scaled,
    y_train
)

best_svr = svr_grid.best_estimator_


# =========================================================
# ALL MODEL PREDICTIONS
# =========================================================

predictions = {

    "Ridge": best_ridge.predict(X_test_scaled),

    "Lasso": best_lasso.predict(X_test_scaled),

    "Decision Tree": tree.predict(X_test),

    "Random Forest": forest.predict(X_test),

    "SVR": best_svr.predict(X_test_scaled)
}


# =========================================================
# MODEL RESULTS
# =========================================================

results = []

for model_name, prediction in predictions.items():

    mse = mean_squared_error(
        y_test,
        prediction
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        prediction
    )

    results.append([
        model_name,
        mse,
        mae,
        rmse,
        r2
    ])


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MSE",
        "MAE",
        "RMSE",
        "R2 Score"
    ]
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🏠 Enter Property Details")

    st.markdown("---")

    selected_model = st.selectbox(
        "Select Model",
        [
            "Ridge",
            "Lasso",
            "Decision Tree",
            "Random Forest",
            "SVR"
        ]
    )

    st.markdown("---")

    area_sqft = st.number_input(
        "House Area (sq.ft)",
        min_value=100.0,
        value=1500.0,
        step=50.0
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        value=2,
        step=1
    )

    location_score = st.number_input(
        "Location Score",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    property_age = st.number_input(
        "Age of Property",
        min_value=0,
        value=10,
        step=1
    )

    distance_city_km = st.number_input(
        "Distance from City (km)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    near_school = st.selectbox(
        "Near School",
        [0, 1]
    )

    near_metro = st.selectbox(
        "Near Metro",
        [0, 1]
    )

    crime_rate_index = st.number_input(
        "Crime Rate Index",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict House Price",
        use_container_width=True
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🏠 Robust Regression Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'House Price Prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)


selected_result = results_df[
    results_df["Model"] == selected_model
].iloc[0]


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">R² Score</div>
            <div class="metric-value">{selected_result["R2 Score"]:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">MAE</div>
            <div class="metric-value">
                ₹ {selected_result["MAE"]:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">
                ₹ {selected_result["RMSE"]:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">MSE</div>
            <div class="metric-value">
                {selected_result["MSE"]:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame(
        [[
            area_sqft,
            bedrooms,
            bathrooms,
            location_score,
            property_age,
            distance_city_km,
            near_school,
            near_metro,
            crime_rate_index
        ]],
        columns=features
    )


    if selected_model == "Ridge":

        input_scaled = scaler.transform(
            input_data
        )

        prediction = best_ridge.predict(
            input_scaled
        )[0]


    elif selected_model == "Lasso":

        input_scaled = scaler.transform(
            input_data
        )

        prediction = best_lasso.predict(
            input_scaled
        )[0]


    elif selected_model == "Decision Tree":

        prediction = tree.predict(
            input_data
        )[0]


    elif selected_model == "Random Forest":

        prediction = forest.predict(
            input_data
        )[0]


    else:

        input_scaled = scaler.transform(
            input_data
        )

        prediction = best_svr.predict(
            input_scaled
        )[0]


    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-title">
                Prediction using {selected_model}
            </div>
            <div class="prediction-price">
                ₹ {prediction:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown(
    '<div class="section-title">🤖 Model Comparison</div>',
    unsafe_allow_html=True
)

display_results = results_df.copy()

display_results["MSE"] = display_results["MSE"].map(
    lambda x: f"{x:,.0f}"
)

display_results["MAE"] = display_results["MAE"].map(
    lambda x: f"{x:,.0f}"
)

display_results["RMSE"] = display_results["RMSE"].map(
    lambda x: f"{x:,.0f}"
)

display_results["R2 Score"] = display_results["R2 Score"].map(
    lambda x: f"{x:.4f}"
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# RMSE GRAPH
# =========================================================

st.markdown(
    '<div class="section-title">📈 Model RMSE Comparison</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    results_df["Model"],
    results_df["RMSE"]
)

ax.set_xlabel("Model")
ax.set_ylabel("RMSE")
ax.set_title("Model Comparison using RMSE")

plt.xticks(rotation=20)
plt.tight_layout()

st.pyplot(fig)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Actual vs Predicted Prices</div>',
    unsafe_allow_html=True
)

selected_prediction = predictions[selected_model]

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.scatter(
    y_test,
    selected_prediction
)

ax2.set_xlabel(
    "Actual House Price"
)

ax2.set_ylabel(
    "Predicted House Price"
)

ax2.set_title(
    f"Actual vs Predicted - {selected_model}"
)

plt.tight_layout()

st.pyplot(fig2)


# =========================================================
# DATASET
# =========================================================

st.markdown(
    '<div class="section-title">📁 Dataset</div>',
    unsafe_allow_html=True
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with info_col2:
    st.metric(
        "Features",
        len(features)
    )

with info_col3:
    st.metric(
        "Target",
        "House Price"
    )


st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)