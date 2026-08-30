import os
import pandas as pd

from data_loader import load_data
from data_cleaning import clean_data
from movement_analysis import calculate_movement
from trajectory_analysis import create_trajectories

from movement_prediction import train_speed_model
from anomaly_detection import detect_movement_anomalies
from movement_forecasting import forecast_movement
from behavior_prediction import predict_behavior

from risk_analysis import (
    calculate_wildlife_risk,
    species_risk_summary,
    conservation_priority,
    risk_distribution,
)

from conservation_intelligence import (
    generate_conservation_intelligence,
    species_conservation_ranking,
    create_risk_distribution,
    species_risk_hotspots,
)

from visualization import (
    species_distribution,
    behavior_distribution,
    location_map,
    species_movement,
    weather_movement,
    vegetation_movement,
    time_of_day_movement,
)

from environmental_analysis import (
    analyze_weather,
    analyze_temperature,
    analyze_vegetation,
    analyze_time_of_day,
)

from movement_patterns import (
    top_moving_animals,
    behavior_movement,
    habitat_preference,
    movement_hotspots,
)


# ======================================================
# CONFIGURATION
# ======================================================

ANALYSIS_OUTPUT_DIR = "outputs/analysis"


# ======================================================
# CREATE OUTPUT DIRECTORIES
# ======================================================

def create_output_directories():
    """Create required output directories."""

    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)


# ======================================================
# SAVE DATAFRAME HELPER
# ======================================================

def save_dataframe(df, filename):
    """Save a DataFrame inside outputs/analysis."""

    if df is None:
        print(f"Skipped: {filename} (no data returned)")
        return

    filepath = os.path.join(
        ANALYSIS_OUTPUT_DIR,
        filename
    )

    df.to_csv(filepath, index=False)

    print(f"Saved: {filepath}")


# ======================================================
# MAIN FUNCTION
# ======================================================

def main():

    print("=" * 60)
    print("       WILDLIFE MOVEMENT ANALYSIS SYSTEM")
    print("=" * 60)

    # ==================================================
    # CREATE OUTPUT DIRECTORIES
    # ==================================================

    create_output_directories()

    # ==================================================
    # 1. LOAD DATASET
    # ==================================================

    print("\n[1] Loading dataset...")

    df = load_data()

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )

    # ==================================================
    # 2. CLEAN DATASET
    # ==================================================

    print("\n[2] Cleaning dataset...")

    df = clean_data(df)

    print(
        f"After cleaning: "
        f"{df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )

    # ==================================================
    # 3. CREATE ANIMAL TRAJECTORIES
    # ==================================================

    print("\n[3] Creating animal trajectories...")

    df = create_trajectories(
        df,
        animals_per_species=10
    )

    print(
        f"Animals created: "
        f"{df['animal_id'].nunique()}"
    )

    # ==================================================
    # 4. MOVEMENT ANALYSIS
    # ==================================================

    print("\n[4] Calculating movement...")

    df = calculate_movement(df)

    print(
        "Distance and speed calculated successfully."
    )

    # ==================================================
    # 5. SPECIES ANALYSIS
    # ==================================================

    print("\n[5] Species Distribution")

    species_counts = df["species"].value_counts()

    print(species_counts)

    # ==================================================
    # 6. BEHAVIOR ANALYSIS
    # ==================================================

    print("\n[6] Behavior Distribution")

    behavior_counts = df["behavior"].value_counts()

    print(behavior_counts)

    # ==================================================
    # 7. SPECIES MOVEMENT SUMMARY
    # ==================================================

    print("\n[7] Species Movement Summary")

    movement_summary = (
        df.groupby("species")
        .agg(
            total_distance_km=(
                "distance_km",
                "sum"
            ),
            average_speed_kmh=(
                "speed_kmh",
                "mean"
            ),
            observations=(
                "animal_id",
                "count"
            )
        )
        .sort_values(
            "total_distance_km",
            ascending=False
        )
    )

    print(movement_summary)

    # ==================================================
    # 8. INDIVIDUAL ANIMAL MOVEMENT
    # ==================================================

    print("\n[8] Individual Animal Movement")

    animal_summary = (
        df.groupby(
            [
                "animal_id",
                "species"
            ]
        )
        .agg(
            total_distance_km=(
                "distance_km",
                "sum"
            ),
            average_speed_kmh=(
                "speed_kmh",
                "mean"
            ),
            observations=(
                "animal_id",
                "count"
            )
        )
        .sort_values(
            "total_distance_km",
            ascending=False
        )
    )

    print(
        animal_summary.head(20)
    )

    # ==================================================
    # 9. ENVIRONMENTAL ANALYSIS
    # ==================================================

    print("\n[9] Environmental Analysis")

    weather_analysis = analyze_weather(df)
    temperature_analysis = analyze_temperature(df)
    vegetation_analysis = analyze_vegetation(df)
    time_analysis = analyze_time_of_day(df)

    print("\nWeather Analysis:")
    print(weather_analysis)

    print("\nTemperature Analysis:")
    print(temperature_analysis)

    print("\nVegetation Analysis:")
    print(vegetation_analysis)

    print("\nTime of Day Analysis:")
    print(time_analysis)

    # ==================================================
    # 10. WILDLIFE MOVEMENT PATTERNS
    # ==================================================

    print("\n[10] Wildlife Movement Patterns")

    top_animals = top_moving_animals(df)

    behavior_movement_analysis = behavior_movement(df)

    habitat_analysis = habitat_preference(df)

    hotspot_analysis = movement_hotspots(df)

    print("\nTop Moving Animals:")
    print(top_animals)

    print("\nBehavior vs Movement:")
    print(behavior_movement_analysis)

    print("\nHabitat Preference:")
    print(habitat_analysis)

    print("\nMovement Hotspots:")
    print(hotspot_analysis.head(20))

    # ==================================================
    # 11. MOVEMENT SPEED PREDICTION
    # ==================================================

    print("\n[11] Movement Speed Prediction")

    (
        speed_model,
        prediction_results,
        mae,
        r2
    ) = train_speed_model(df)

    print("\nModel Performance:")

    print(
        f"Mean Absolute Error: {mae:.4f}"
    )

    print(
        f"R2 Score: {r2:.4f}"
    )

    print("\nPrediction Results:")

    print(
        prediction_results.head(10)
    )

    # ==================================================
    # 12. MOVEMENT ANOMALY DETECTION
    # ==================================================

    print("\n[12] Movement Anomaly Detection")

    (
        anomaly_data,
        anomalies
    ) = detect_movement_anomalies(df)

    print(
        f"\nAnomalies detected: "
        f"{len(anomalies)}"
    )

    if len(anomalies) > 0:

        print("\nAnomaly Preview:")

        print(
            anomalies.head(10)
        )

    else:

        print(
            "No significant movement anomalies detected."
        )

    # ==================================================
    # 13. MOVEMENT FORECASTING
    # ==================================================

    print(
        "\n[13] Wildlife Movement Forecasting"
    )

    (
        forecast_model,
        forecast_results,
        forecast_mae,
        forecast_r2
    ) = forecast_movement(df)

    print(
        f"\nForecast MAE: "
        f"{forecast_mae:.4f}"
    )

    print(
        f"Forecast R2 Score: "
        f"{forecast_r2:.4f}"
    )

    print("\nForecast Results:")

    print(
        forecast_results.head(10)
    )

    # ==================================================
    # 14. BEHAVIOR PREDICTION
    # ==================================================

    print("\n[14] Behavior Prediction")

    (
        behavior_model,
        behavior_predictions,
        behavior_accuracy
    ) = predict_behavior(df)

    print(
        f"Behavior prediction accuracy: "
        f"{behavior_accuracy:.4f}"
    )

    print("\nBehavior Prediction Results:")

    print(
        behavior_predictions.head(10)
    )

    # ==================================================
    # 15. WILDLIFE RISK ANALYSIS
    # ==================================================

    print(
        "\n[15] Wildlife Risk & Conservation Intelligence"
    )

    # --------------------------------------------------
    # Calculate Wildlife Risk
    # --------------------------------------------------

    print("\nCalculating wildlife risk...")

    risk_data = calculate_wildlife_risk(df)

    print("Risk calculation completed.")

    # --------------------------------------------------
    # Risk Distribution
    # --------------------------------------------------

    print("\nRisk Distribution:")

    risk_counts = (
        risk_data["risk_level"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Moderate",
                "High",
                "Critical"
            ],
            fill_value=0
        )
    )

    print(risk_counts)

    # --------------------------------------------------
    # Species Risk Summary
    # --------------------------------------------------

    print("\nSpecies Risk Summary:")

    species_risk = species_risk_summary(
        risk_data
    )

    print(species_risk)

    # --------------------------------------------------
    # Conservation Priority
    # --------------------------------------------------

    print("\nConservation Priority:")

    priority_animals = conservation_priority(
        risk_data
    )

    priority_columns = [
        "animal_id",
        "species",
        "risk_score",
        "risk_level",
        "conservation_priority",
        "risk_reason"
    ]

    available_priority_columns = [
        column
        for column in priority_columns
        if column in priority_animals.columns
    ]

    print(
        priority_animals[
            available_priority_columns
        ].head(20)
    )

    # --------------------------------------------------
    # High/Critical Count
    # --------------------------------------------------

    high_critical_count = risk_data[
        risk_data["risk_level"].isin(
            ["High", "Critical"]
        )
    ].shape[0]

    print(
        f"\nHigh/Critical observations: "
        f"{high_critical_count}"
    )

    # ==================================================
    # 16. SAVE P1-P10 ANALYSIS RESULTS
    # ==================================================

    print("\n[16] Saving analysis results...")

    # --------------------------------------------------
    # Complete processed dataset
    # --------------------------------------------------

    save_dataframe(
        df,
        "movement_data.csv"
    )

    # --------------------------------------------------
    # Species summary
    # --------------------------------------------------

    save_dataframe(
        movement_summary.reset_index(),
        "species_movement_summary.csv"
    )

    # --------------------------------------------------
    # Individual animal summary
    # --------------------------------------------------

    save_dataframe(
        animal_summary.reset_index(),
        "animal_movement_summary.csv"
    )

    # --------------------------------------------------
    # Environmental analysis
    # --------------------------------------------------

    save_dataframe(
        weather_analysis,
        "weather_analysis.csv"
    )

    save_dataframe(
        temperature_analysis,
        "temperature_analysis.csv"
    )

    save_dataframe(
        vegetation_analysis,
        "vegetation_analysis.csv"
    )

    save_dataframe(
        time_analysis,
        "time_of_day_analysis.csv"
    )

    # --------------------------------------------------
    # Movement pattern analysis
    # --------------------------------------------------

    save_dataframe(
        top_animals,
        "top_moving_animals.csv"
    )

    save_dataframe(
        behavior_movement_analysis,
        "behavior_movement.csv"
    )

    save_dataframe(
        habitat_analysis,
        "habitat_preference.csv"
    )

    save_dataframe(
        hotspot_analysis,
        "movement_hotspots.csv"
    )

    # --------------------------------------------------
    # Speed prediction
    # --------------------------------------------------

    save_dataframe(
        prediction_results,
        "movement_predictions.csv"
    )

    model_performance = pd.DataFrame(
        {
            "MAE": [mae],
            "R2_Score": [r2]
        }
    )

    save_dataframe(
        model_performance,
        "model_performance.csv"
    )

    # --------------------------------------------------
    # Anomaly results
    # --------------------------------------------------

    save_dataframe(
        anomaly_data,
        "movement_anomalies.csv"
    )

    # --------------------------------------------------
    # Movement forecasting
    # --------------------------------------------------

    save_dataframe(
        forecast_results,
        "movement_forecast.csv"
    )

    forecast_performance = pd.DataFrame(
        {
            "MAE": [forecast_mae],
            "R2_Score": [forecast_r2]
        }
    )

    save_dataframe(
        forecast_performance,
        "forecast_performance.csv"
    )

    # --------------------------------------------------
    # Behavior prediction
    # --------------------------------------------------

    save_dataframe(
        behavior_predictions,
        "behavior_predictions.csv"
    )

    behavior_performance = pd.DataFrame(
        {
            "Accuracy": [behavior_accuracy]
        }
    )

    save_dataframe(
        behavior_performance,
        "behavior_model_performance.csv"
    )

    # ==================================================
    # P10 RISK RESULTS
    # ==================================================

    print("\nSaving P10 risk analysis results...")

    # Complete risk data
    save_dataframe(
        risk_data,
        "wildlife_risk_analysis.csv"
    )

    # Species risk summary
    save_dataframe(
        species_risk.reset_index(),
        "species_risk_summary.csv"
    )

    # Conservation priority
    save_dataframe(
        priority_animals,
        "conservation_priority.csv"
    )

    # Risk distribution
    risk_distribution_data = (
        risk_data["risk_level"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Moderate",
                "High",
                "Critical"
            ],
            fill_value=0
        )
        .reset_index()
    )

    risk_distribution_data.columns = [
        "risk_level",
        "observations"
    ]

    save_dataframe(
        risk_distribution_data,
        "risk_distribution.csv"
    )

    # ==================================================
    # 17. P11 - CONSERVATION INTELLIGENCE
    # ==================================================

    print("\n" + "=" * 60)
    print("       P11 - CONSERVATION INTELLIGENCE")
    print("=" * 60)

    # --------------------------------------------------
    # Generate Conservation Intelligence
    # --------------------------------------------------

    print("\nGenerating conservation intelligence...")

    conservation_data = generate_conservation_intelligence(
        df,
        risk_data,
        species_risk,
        priority_animals,
        anomalies
    )

    print("\nConservation Intelligence:")

    print(
        conservation_data.to_string(index=False)
    )

    # --------------------------------------------------
    # Species Conservation Ranking
    # --------------------------------------------------

    print(
        "\nGenerating species conservation ranking..."
    )

    conservation_ranking = species_conservation_ranking(
        species_risk
    )

    print("\nSpecies Conservation Ranking:")

    print(
        conservation_ranking.to_string(index=False)
    )

    # --------------------------------------------------
    # Conservation Risk Distribution
    # --------------------------------------------------

    print(
        "\nGenerating conservation risk distribution..."
    )

    conservation_risk_distribution = create_risk_distribution(
        risk_data
    )

    print("\nConservation Risk Distribution:")

    print(
        conservation_risk_distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Species Risk Hotspots
    # --------------------------------------------------

    print(
        "\nGenerating species risk hotspots..."
    )

    risk_hotspots = species_risk_hotspots(
        risk_data,
        top_n=10
    )

    print("\nSpecies Risk Hotspots:")

    print(
        risk_hotspots.to_string(index=False)
    )

    # ==================================================
    # SAVE P11 RESULTS
    # ==================================================

    print(
        "\nSaving P11 conservation intelligence results..."
    )

    # --------------------------------------------------
    # Conservation Intelligence
    # --------------------------------------------------

    save_dataframe(
        conservation_data,
        "conservation_intelligence.csv"
    )

    # --------------------------------------------------
    # Species Conservation Ranking
    # --------------------------------------------------

    save_dataframe(
        conservation_ranking,
        "species_conservation_ranking.csv"
    )

    # --------------------------------------------------
    # Conservation Risk Distribution
    # --------------------------------------------------

    save_dataframe(
        conservation_risk_distribution,
        "conservation_risk_distribution.csv"
    )

    # --------------------------------------------------
    # Species Risk Hotspots
    # --------------------------------------------------

    save_dataframe(
        risk_hotspots,
        "species_risk_hotspots.csv"
    )

    print(
        "\nP11 conservation intelligence completed successfully."
    )

    # ==================================================
    # 18. GENERATE VISUALIZATIONS
    # ==================================================

    print("\n[18] Creating visualizations...")

    # --------------------------------------------------
    # Basic charts
    # --------------------------------------------------

    print(
        "Creating species distribution chart..."
    )

    species_distribution(df)

    print(
        "Creating behavior distribution chart..."
    )

    behavior_distribution(df)

    print(
        "Creating location map..."
    )

    location_map(df)

    # --------------------------------------------------
    # Movement charts
    # --------------------------------------------------

    print(
        "Creating species movement chart..."
    )

    species_movement(df)

    print(
        "Creating weather movement chart..."
    )

    weather_movement(df)

    print(
        "Creating vegetation movement chart..."
    )

    vegetation_movement(df)

    print(
        "Creating time-of-day movement chart..."
    )

    time_of_day_movement(df)

    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n" + "=" * 60)

    print(
        "       ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        f"\nAnalysis files saved in: "
        f"{ANALYSIS_OUTPUT_DIR}"
    )

    print(
        "Charts saved in: outputs/charts"
    )

    print(
        "Maps saved in: outputs/maps"
    )

    print(
        "\nP10: Wildlife Risk Analysis - COMPLETED"
    )

    print(
        "P11: Conservation Intelligence - COMPLETED"
    )

    print("\nThank you.")


# ======================================================
# PROGRAM ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()