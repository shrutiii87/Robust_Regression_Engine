import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Robust Regression Engine",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: white;
}

label {
    color: white !important;
}

.stButton > button {
    width: 100%;
    height: 42px;
    border-radius: 8px;
    font-size: 15px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #161b22;
    border: 1px solid #30363d;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🏠 Robust Regression Engine")

st.write(
    "House Price Prediction using Regression Models"
)


# =========================================================
# LOAD DATASET
# =========================================================

file_name = (
    "Advanced_Regression_HousePrice_Dataset_3800 "
    "- Advanced_Regression_HousePrice_Dataset_3800.csv.csv"
)


@st.cache_data
def load_data():

    data = pd.read_csv(file_name)

    return data


df = load_data()


# =========================================================
# FEATURES AND TARGET
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


# =========================================================
# CHECK COLUMNS
# =========================================================

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error("Required columns are missing from the dataset.")

    st.write(missing_columns)

    st.stop()


# =========================================================
# PREPARE DATA
# =========================================================

data = df[required_columns].dropna()

X = data[features]

y = data[target]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# TRAIN ALL MODELS
# =========================================================

@st.cache_resource
def train_models(X_train, y_train):

    # -----------------------------------------------------
    # STANDARD SCALER
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)


    # -----------------------------------------------------
    # RIDGE
    # -----------------------------------------------------

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )


    # -----------------------------------------------------
    # LASSO
    # -----------------------------------------------------

    lasso = Lasso(
        alpha=100
    )

    lasso.fit(
        X_train_scaled,
        y_train
    )


    # -----------------------------------------------------
    # DECISION TREE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # SVR
    # -----------------------------------------------------

    svr = SVR(
        kernel="rbf",
        C=10,
        gamma=0.1,
        epsilon=0.1
    )

    svr.fit(
        X_train_scaled,
        y_train
    )


    models = {
        "Ridge": ridge,
        "Lasso": lasso,
        "Decision Tree": tree,
        "Random Forest": forest,
        "SVR": svr
    }

    return models, scaler


models, scaler = train_models(
    X_train,
    y_train
)


# =========================================================
# TEST DATA SCALING
# =========================================================

X_test_scaled = scaler.transform(X_test)


# =========================================================
# MODEL PREDICTIONS
# =========================================================

predictions = {}


predictions["Ridge"] = models["Ridge"].predict(
    X_test_scaled
)


predictions["Lasso"] = models["Lasso"].predict(
    X_test_scaled
)


predictions["Decision Tree"] = models["Decision Tree"].predict(
    X_test
)


predictions["Random Forest"] = models["Random Forest"].predict(
    X_test
)


predictions["SVR"] = models["SVR"].predict(
    X_test_scaled
)


# =========================================================
# MODEL METRICS
# =========================================================

model_metrics = {}


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

    model_metrics[model_name] = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# SIDEBAR FORM
# =========================================================

with st.sidebar.form("prediction_form"):

    st.title("🏠 Property Details")

    st.write(
        "Enter property information:"
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # PROPERTY INPUTS
    # -----------------------------------------------------

    area_sqft = st.number_input(
        "Area (sqft)",
        min_value=0.0,
        value=float(X["area_sqft"].median()),
        step=1.0
    )


    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        value=int(X["bedrooms"].median()),
        step=1
    )


    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0,
        value=int(X["bathrooms"].median()),
        step=1
    )


    location_score = st.number_input(
        "Location Score",
        min_value=0.0,
        value=float(X["location_score"].median()),
        step=0.1
    )


    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        value=int(X["property_age"].median()),
        step=1
    )


    distance_city_km = st.number_input(
        "Distance from City (km)",
        min_value=0.0,
        value=float(X["distance_city_km"].median()),
        step=0.1
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
        value=float(X["crime_rate_index"].median()),
        step=0.1
    )


    # -----------------------------------------------------
    # SUBMIT BUTTON
    # -----------------------------------------------------

    predict_button = st.form_submit_button(
        "Predict House Price"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Dataset Rows",
        len(data)
    )


with col2:

    st.metric(
        "Training Rows",
        len(X_train)
    )


with col3:

    st.metric(
        "Testing Rows",
        len(X_test)
    )


# =========================================================
# SELECTED MODEL PERFORMANCE
# =========================================================

metrics = model_metrics[selected_model]


st.subheader(
    f"{selected_model} Performance"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )


with col2:

    st.metric(
        "MAE",
        f"₹{metrics['MAE']:,.0f}"
    )


with col3:

    st.metric(
        "RMSE",
        f"₹{metrics['RMSE']:,.0f}"
    )


with col4:

    st.metric(
        "MSE",
        f"{metrics['MSE']:,.0f}"
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


    # -----------------------------------------------------
    # SCALE INPUT FOR LINEAR + SVR MODELS
    # -----------------------------------------------------

    if selected_model in [
        "Ridge",
        "Lasso",
        "SVR"
    ]:

        input_scaled = scaler.transform(
            input_data
        )

        prediction = models[selected_model].predict(
            input_scaled
        )[0]


    # -----------------------------------------------------
    # TREE MODELS DON'T NEED SCALING
    # -----------------------------------------------------

    else:

        prediction = models[selected_model].predict(
            input_data
        )[0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.subheader("Prediction")

    st.markdown(
        f"""
        <div class="result-box">

        <h3>Selected Model</h3>

        <h2>{selected_model}</h2>

        <h3>Predicted House Price</h3>

        <h1>₹{prediction:,.0f}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL DETAILS
# =========================================================

st.subheader("Model Details")


if selected_model == "Ridge":

    st.write("**Model:** Ridge Regression")

    st.write("**Alpha:** 1.0")


elif selected_model == "Lasso":

    st.write("**Model:** Lasso Regression")

    st.write("**Alpha:** 100")


elif selected_model == "Decision Tree":

    st.write("**Model:** Decision Tree Regression")

    st.write("**Max Depth:** 5")

    st.write("**Min Samples Split:** 10")

    st.write("**Min Samples Leaf:** 5")


elif selected_model == "Random Forest":

    st.write("**Model:** Random Forest Regression")

    st.write("**Number of Trees:** 100")

    st.write("**Max Depth:** 5")

    st.write("**Min Samples Split:** 10")

    st.write("**Min Samples Leaf:** 5")


elif selected_model == "SVR":

    st.write("**Model:** Support Vector Regression")

    st.write("**Kernel:** RBF")

    st.write("**C:** 10")

    st.write("**Gamma:** 0.1")

    st.write("**Epsilon:** 0.1")


# =========================================================
# FEATURES
# =========================================================

st.write("**Features used:**")

st.write(features)
