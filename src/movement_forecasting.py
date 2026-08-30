import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_movement(df):

    print("\n" + "=" * 50)
    print("WILDLIFE MOVEMENT FORECASTING")
    print("=" * 50)

    data = df.copy()

    # --------------------------------------------------
    # 1. Prepare timestamp
    # --------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data = data.sort_values(
        ["animal_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Create historical movement features
    # --------------------------------------------------

    data["previous_speed_kmh"] = (
        data.groupby("animal_id")["speed_kmh"]
        .shift(1)
    )

    data["previous_distance_km"] = (
        data.groupby("animal_id")["distance_km"]
        .shift(1)
    )

    data["rolling_speed_kmh"] = (
        data.groupby("animal_id")["speed_kmh"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )

    data["rolling_distance_km"] = (
        data.groupby("animal_id")["distance_km"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    # --------------------------------------------------
    # 3. Remove rows without historical information
    # --------------------------------------------------

    features = [
        "previous_speed_kmh",
        "previous_distance_km",
        "rolling_speed_kmh",
        "rolling_distance_km",
        "hour",
        "day_of_week",
        "latitude",
        "longitude",
    ]

    data = data.dropna(
        subset=features + ["speed_kmh"]
    )

    # --------------------------------------------------
    # 4. Prepare X and y
    # --------------------------------------------------

    X = data[features].copy()

    y = data["speed_kmh"].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # --------------------------------------------------
    # 5. Time-based train/test split
    # --------------------------------------------------

    split_index = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(
        f"\nTraining observations: "
        f"{len(X_train)}"
    )

    print(
        f"Testing observations: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------
    # 6. Create model
    # --------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    print(
        "\nTraining forecasting model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Forecasting model trained."
    )

    # --------------------------------------------------
    # 7. Predict
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # 8. Evaluate
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nForecast Performance")
    print("-" * 30)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # --------------------------------------------------
    # 9. Create forecast results
    # --------------------------------------------------

    forecast_results = data.iloc[
        split_index:
    ][
        [
            "animal_id",
            "species",
            "timestamp",
            "latitude",
            "longitude",
            "speed_kmh",
        ]
    ].copy()

    forecast_results[
        "predicted_speed_kmh"
    ] = predictions

    forecast_results[
        "speed_difference"
    ] = (
        forecast_results["speed_kmh"]
        - forecast_results["predicted_speed_kmh"]
    )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    os.makedirs(
        "outputs/analysis",
        exist_ok=True
    )

    forecast_results.to_csv(
        "outputs/analysis/"
        "movement_forecast.csv",
        index=False
    )

    print(
        "\nSaved: "
        "outputs/analysis/"
        "movement_forecast.csv"
    )

    # --------------------------------------------------
    # 11. Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        "outputs/analysis/"
        "forecast_feature_importance.csv",
        index=False
    )

    print(
        "Saved: "
        "outputs/analysis/"
        "forecast_feature_importance.csv"
    )

    # --------------------------------------------------
    # 12. Actual vs predicted graph
    # --------------------------------------------------

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        y_test.values,
        label="Actual Speed"
    )

    plt.plot(
        predictions,
        label="Forecast Speed"
    )

    plt.xlabel(
        "Observation"
    )

    plt.ylabel(
        "Speed (km/h)"
    )

    plt.title(
        "Wildlife Movement Speed Forecast"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/charts/"
        "movement_forecast.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/"
        "movement_forecast.png"
    )

    return (
        model,
        forecast_results,
        mae,
        r2
    )