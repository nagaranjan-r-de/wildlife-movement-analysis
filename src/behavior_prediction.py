import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def predict_behavior(df):

    print("\n" + "=" * 50)
    print("P9 - BEHAVIOR PREDICTION")
    print("=" * 50)

    data = df.copy()

    # --------------------------------------------------
    # Create target
    # --------------------------------------------------

    target = "behavior"

    # Remove rows without behavior
    data = data.dropna(subset=[target])

    # --------------------------------------------------
    # Select useful features
    # --------------------------------------------------

    feature_columns = [
        "latitude",
        "longitude",
        "temperature",
        "distance_km",
        "speed_kmh",
    ]

    # Add environmental categorical features if available
    categorical_columns = []

    for column in [
        "species",
        "weather",
        "vegetation_type",
        "time_of_day",
    ]:
        if column in data.columns:
            categorical_columns.append(column)

    # --------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------

    X = data[feature_columns + categorical_columns].copy()

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            X[column].astype(str)
        )

        encoders[column] = encoder

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    valid_rows = X.notna().all(axis=1)

    X = X[valid_rows]
    y = data.loc[X.index, target]

    # --------------------------------------------------
    # Encode target
    # --------------------------------------------------

    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(
        y.astype(str)
    )

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    print(
        f"\nBehavior Prediction Accuracy: "
        f"{accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_encoder.classes_,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Prediction results
    # --------------------------------------------------

    prediction_results = data.loc[
        X_test.index
    ].copy()

    prediction_results[
        "actual_behavior"
    ] = target_encoder.inverse_transform(
        y_test
    )

    prediction_results[
        "predicted_behavior"
    ] = target_encoder.inverse_transform(
        y_pred
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        "outputs/analysis",
        exist_ok=True,
    )

    prediction_results.to_csv(
        "outputs/analysis/"
        "behavior_predictions.csv",
        index=False,
    )

    print(
        "\nSaved: "
        "outputs/analysis/"
        "behavior_predictions.csv"
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(cm)

    plt.title(
        "Wildlife Behavior Prediction"
    )

    plt.xlabel(
        "Predicted Behavior"
    )

    plt.ylabel(
        "Actual Behavior"
    )

    plt.xticks(
        range(len(target_encoder.classes_)),
        target_encoder.classes_,
        rotation=45,
    )

    plt.yticks(
        range(len(target_encoder.classes_)),
        target_encoder.classes_,
    )

    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    os.makedirs(
        "outputs/charts",
        exist_ok=True,
    )

    plt.savefig(
        "outputs/charts/"
        "behavior_prediction_confusion_matrix.png"
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/"
        "behavior_prediction_confusion_matrix.png"
    )

    return (
        model,
        prediction_results,
        accuracy,
    )
