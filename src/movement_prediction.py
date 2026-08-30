import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_speed_model(df):

    print("\n" + "=" * 50)
    print("MOVEMENT SPEED PREDICTION")
    print("=" * 50)

    data = df.copy()

    # --------------------------------------------------
    # 1. Prepare timestamp
    # --------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 2. Create time-based numerical features
    # --------------------------------------------------

    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    # --------------------------------------------------
    # 3. Select features
    # --------------------------------------------------

    feature_columns = [
        "latitude",
        "longitude",
        "temperature",
        "hour",
        "day",
        "month",
        "day_of_week"
    ]

    # Add weather if available
    if "weather" in data.columns:
        data = pd.get_dummies(
            data,
            columns=["weather"],
            dtype=int
        )

        weather_columns = [
            col for col in data.columns
            if col.startswith("weather_")
        ]

        feature_columns.extend(weather_columns)

    # Add vegetation if available
    if "vegetation_type" in data.columns:
        data = pd.get_dummies(
            data,
            columns=["vegetation_type"],
            dtype=int
        )

        vegetation_columns = [
            col for col in data.columns
            if col.startswith("vegetation_type_")
        ]

        feature_columns.extend(vegetation_columns)

    # --------------------------------------------------
    # 4. Check required columns
    # --------------------------------------------------

    required_columns = [
        "latitude",
        "longitude",
        "temperature",
        "speed_kmh"
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------
    # 5. Remove missing values
    # --------------------------------------------------

    data = data.dropna(
        subset=[
            "latitude",
            "longitude",
            "temperature",
            "speed_kmh",
            "hour",
            "day",
            "month",
            "day_of_week"
        ]
    )

    # --------------------------------------------------
    # 6. Build X and y
    # --------------------------------------------------

    X = data[feature_columns]
    y = data["speed_kmh"]

    # Make sure every feature is numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # --------------------------------------------------
    # 7. Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # 8. Create Random Forest model
    # --------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------
    # 9. Train model
    # --------------------------------------------------

    print("\nTraining Random Forest model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # --------------------------------------------------
    # 10. Prediction
    # --------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 11. Evaluation
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

    print("\nModel Performance")
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
    # 12. Save predictions
    # --------------------------------------------------

    os.makedirs(
        "outputs/analysis",
        exist_ok=True
    )

    prediction_results = pd.DataFrame({
        "actual_speed_kmh": y_test.values,
        "predicted_speed_kmh": predictions
    })

    prediction_results.to_csv(
        "outputs/analysis/"
        "speed_predictions.csv",
        index=False
    )

    print(
        "\nSaved: "
        "outputs/analysis/speed_predictions.csv"
    )

    # --------------------------------------------------
    # 13. Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance")
    print(importance)

    importance.to_csv(
        "outputs/analysis/"
        "speed_feature_importance.csv",
        index=False
    )

    print(
        "Saved: "
        "outputs/analysis/"
        "speed_feature_importance.csv"
    )

    # --------------------------------------------------
    # 14. Actual vs predicted graph
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    plt.xlabel(
        "Actual Speed (km/h)"
    )

    plt.ylabel(
        "Predicted Speed (km/h)"
    )

    plt.title(
        "Actual vs Predicted Wildlife Movement Speed"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/charts/"
        "speed_prediction.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/speed_prediction.png"
    )

    # --------------------------------------------------
    # 15. Return model and results
    # --------------------------------------------------

    return model, prediction_results, mae, r2