import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest


def detect_movement_anomalies(df):

    print("\n" + "=" * 50)
    print("MOVEMENT ANOMALY DETECTION")
    print("=" * 50)

    data = df.copy()

    # Features used for anomaly detection
    features = [
        "distance_km",
        "speed_kmh",
        "latitude",
        "longitude",
    ]

    # Check columns
    missing = [
        column
        for column in features
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Remove missing values
    model_data = data[features].copy()

    model_data = model_data.fillna(
        model_data.median()
    )

    # Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(
        model_data
    )

    # -1 = anomaly
    #  1 = normal
    data["anomaly"] = predictions

    data["movement_status"] = data[
        "anomaly"
    ].map({
        1: "Normal",
        -1: "Anomaly"
    })

    # Get anomalies
    anomalies = data[
        data["anomaly"] == -1
    ].copy()

    print(
        f"\nTotal observations: {len(data)}"
    )

    print(
        f"Normal observations: "
        f"{(data['anomaly'] == 1).sum()}"
    )

    print(
        f"Anomalous observations: "
        f"{(data['anomaly'] == -1).sum()}"
    )

    print("\nTop movement anomalies:")

    print(
        anomalies[
            [
                "animal_id",
                "species",
                "timestamp",
                "distance_km",
                "speed_kmh",
                "latitude",
                "longitude",
            ]
        ]
        .sort_values(
            "speed_kmh",
            ascending=False
        )
        .head(20)
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        "outputs/analysis",
        exist_ok=True
    )

    anomalies.to_csv(
        "outputs/analysis/"
        "movement_anomalies.csv",
        index=False
    )

    print(
        "\nSaved: "
        "outputs/analysis/"
        "movement_anomalies.csv"
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    normal = data[
        data["anomaly"] == 1
    ]

    abnormal = data[
        data["anomaly"] == -1
    ]

    plt.scatter(
        normal["distance_km"],
        normal["speed_kmh"],
        alpha=0.5,
        label="Normal"
    )

    plt.scatter(
        abnormal["distance_km"],
        abnormal["speed_kmh"],
        alpha=0.8,
        label="Anomaly"
    )

    plt.xlabel(
        "Distance (km)"
    )

    plt.ylabel(
        "Speed (km/h)"
    )

    plt.title(
        "Wildlife Movement Anomaly Detection"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/charts/"
        "movement_anomalies.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/"
        "movement_anomalies.png"
    )

    return data, anomalies