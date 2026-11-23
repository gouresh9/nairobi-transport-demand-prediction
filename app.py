from datetime import date, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path(__file__).parent / "data" / "train_revised.csv"

CATEGORICAL = ["travel_from", "travel_to", "car_type"]
NUMERIC = [
    "max_capacity",
    "travel_month",
    "travel_weekday",
    "departure_hour",
]


@st.cache_resource
def load_data_and_train():
    raw = pd.read_csv(DATA_FILE)

    required = [
        "ride_id",
        "seat_number",
        "travel_date",
        "travel_time",
        "travel_from",
        "travel_to",
        "car_type",
        "max_capacity",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing these columns: {missing}")

    # Each CSV row is a seat record. Count distinct seats for each ride.
    rides = raw.groupby("ride_id", as_index=False).agg(
        travel_date=("travel_date", "first"),
        travel_time=("travel_time", "first"),
        travel_from=("travel_from", "first"),
        travel_to=("travel_to", "first"),
        car_type=("car_type", "first"),
        max_capacity=("max_capacity", "first"),
        seats_sold=("seat_number", "nunique"),
    )

    rides["max_capacity"] = pd.to_numeric(rides["max_capacity"])
    parsed_dates = pd.to_datetime(
        rides["travel_date"],
        format="%d-%m-%y",
        errors="raise",
    )
    rides["travel_month"] = parsed_dates.dt.month
    rides["travel_weekday"] = parsed_dates.dt.dayofweek
    rides["departure_hour"] = (
        rides["travel_time"].astype(str).str.split(":").str[0].astype(int)
    )

    features = rides[CATEGORICAL + NUMERIC]
    target = rides["seats_sold"]
    capacity = rides["max_capacity"]

    # Encode route and vehicle categories, then fit Linear Regression.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL,
            ),
            ("numbers", "passthrough", NUMERIC),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test, _, capacity_test = train_test_split(
        features,
        target,
        capacity,
        test_size=0.20,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, capacity_test.to_numpy())

    baseline_predictions = np.full(len(y_test), y_train.mean())
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "Baseline MAE": mean_absolute_error(y_test, baseline_predictions),
        "Baseline RMSE": np.sqrt(
            mean_squared_error(y_test, baseline_predictions)
        ),
        "test_count": len(y_test),
    }

    return raw, rides, model, metrics, parsed_dates


st.set_page_config(
    page_title="Transport Demand Prediction",
    page_icon="🚌",
    layout="wide",
)

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Put train_revised.csv inside the data folder."
    )
    st.stop()

try:
    raw, rides, model, metrics, parsed_dates = load_data_and_train()
except Exception as error:
    st.error(f"Could not load or prepare the data: {error}")
    st.stop()

st.title("Transport Demand Prediction Using Regression")
st.caption("Nairobi transport data • Student project")

page = st.sidebar.radio(
    "Choose a page",
    ["Project Overview", "Data & Results", "Predict Seats"],
)

if page == "Project Overview":
    st.header("Project objective")
    st.write(
        "Estimate the number of seats sold for a ride using its route, "
        "travel date, and departure time."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Booking records", f"{len(raw):,}")
    col2.metric("Unique rides", f"{len(rides):,}")
    col3.metric("Dataset columns", len(raw.columns))

    st.subheader("How the model works")
    st.write(
        "The app groups seat records by ride and counts distinct seat numbers "
        "to calculate seats sold. Linear Regression then uses route, vehicle, "
        "capacity, month, weekday, and departure hour."
    )
    st.caption(
        "The report says 14 routes but lists 16 origin towns. Verify the "
        "route count against the dataset; this app does not reconcile it."
    )

elif page == "Data & Results":
    st.header("Data and model results")
    st.write(
        f"The training file has {len(raw):,} booking records, grouped into "
        f"{len(rides):,} rides for this analysis."
    )

    st.subheader("Seats sold per ride")
    seat_bands = pd.cut(
        rides["seats_sold"],
        bins=[0, 5, 10, 15, 20, 30, 49],
        labels=["1–5", "6–10", "11–15", "16–20", "21–30", "31–49"],
        include_lowest=True,
    )
    counts = seat_bands.value_counts(sort=False)

    fig1, ax1 = plt.subplots()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax1)
    ax1.set_xlabel("Seats sold")
    ax1.set_ylabel("Number of rides")
    ax1.set_title("Ride count by seats sold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Held-out test results")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.2f} seats")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f} seats")
    col3.metric("R²", f"{metrics['R2']:.3f}")

    st.caption(
        f"Random 80/20 split by ride; {metrics['test_count']:,} rides in the "
        "test set. Lower MAE and RMSE mean smaller prediction errors."
    )

    comparison = pd.DataFrame(
        {
            "Method": ["Linear Regression", "Mean baseline"],
            "MAE": [metrics["MAE"], metrics["Baseline MAE"]],
            "RMSE": [metrics["RMSE"], metrics["Baseline RMSE"]],
        }
    ).melt(id_vars="Method", var_name="Metric", value_name="Error in seats")

    fig2, ax2 = plt.subplots()
    sns.barplot(
        data=comparison,
        x="Metric",
        y="Error in seats",
        hue="Method",
        ax=ax2,
    )
    ax2.set_title("Model error compared with a mean baseline")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

else:
    st.header("Predict seats for a ride")
    st.write("Choose trip details from values available in the training data.")

    origins = sorted(raw["travel_from"].dropna().astype(str).unique())
    destinations = sorted(raw["travel_to"].dropna().astype(str).unique())
    vehicles = sorted(raw["car_type"].dropna().astype(str).unique())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        origin = col1.selectbox("Origin", origins)
        destination = col2.selectbox("Destination", destinations)
        vehicle = col1.selectbox("Vehicle type", vehicles)

        capacities = sorted(
            rides.loc[
                rides["car_type"].astype(str) == vehicle,
                "max_capacity",
            ].astype(int).unique()
        )
        selected_capacity = col2.selectbox(
            "Maximum seat capacity",
            capacities,
        )

        latest_training_date = parsed_dates.max().date()
        travel_day = col1.date_input(
            "Travel date",
            value=latest_training_date,
        )
        departure = col2.time_input(
            "Departure time",
            value=time(8, 0),
        )

        submitted = st.form_submit_button("Predict seats sold")

    if submitted:
        new_ride = pd.DataFrame(
            [
                {
                    "travel_from": origin,
                    "travel_to": destination,
                    "car_type": vehicle,
                    "max_capacity": float(selected_capacity),
                    "travel_month": travel_day.month,
                    "travel_weekday": travel_day.weekday(),
                    "departure_hour": departure.hour,
                }
            ]
        )

        estimate = float(model.predict(new_ride)[0])
        estimate = float(np.clip(estimate, 0, selected_capacity))

        st.success(
            f"Estimated seats sold: {estimate:.1f} "
            f"out of {selected_capacity} seats"
        )
        st.caption(
            "This is an estimate from historical training data, "
            "not a guaranteed sales count."
        )