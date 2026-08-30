import numpy as np
import pandas as pd


# ======================================================
# WILDLIFE RISK ANALYSIS
# ======================================================

def calculate_wildlife_risk(df):
    """
    Calculate an interpretable wildlife movement risk score.

    Risk factors:
    - Movement anomaly
    - Unusually high speed
    - Large movement distance
    - Environmental conditions
    - Risk-related behavior
    """

    result = df.copy()

    # --------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------

    required_columns = [
        "species",
        "behavior",
        "speed_kmh",
        "distance_km",
        "weather",
        "vegetation_type",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------

    result["speed_kmh"] = pd.to_numeric(
        result["speed_kmh"],
        errors="coerce"
    ).fillna(0)

    result["distance_km"] = pd.to_numeric(
        result["distance_km"],
        errors="coerce"
    ).fillna(0)

    # ==================================================
    # 1. MOVEMENT RISK
    # ==================================================

    speed_95 = result["speed_kmh"].quantile(0.95)
    distance_95 = result["distance_km"].quantile(0.95)

    result["speed_risk"] = np.where(
        result["speed_kmh"] >= speed_95,
        20,
        0
    )

    result["distance_risk"] = np.where(
        result["distance_km"] >= distance_95,
        20,
        0
    )

    # ==================================================
    # 2. ANOMALY RISK
    # ==================================================

    if "anomaly" in result.columns:

        result["anomaly_risk"] = np.where(
            result["anomaly"] == -1,
            30,
            0
        )

    elif "movement_status" in result.columns:

        result["anomaly_risk"] = np.where(
            result["movement_status"].astype(str).str.lower()
            == "anomaly",
            30,
            0
        )

    else:

        result["anomaly_risk"] = 0

    # ==================================================
    # 3. WEATHER RISK
    # ==================================================

    weather_risk_map = {
        "Rainy": 5,
        "Windy": 5,
        "Foggy": 8,
        "Cloudy": 2,
        "Sunny": 0,
    }

    result["weather_risk"] = (
        result["weather"]
        .map(weather_risk_map)
        .fillna(0)
    )

    # ==================================================
    # 4. HABITAT RISK
    # ==================================================

    habitat_risk_map = {
        "Wetland": 5,
        "Forest": 3,
        "Savanna": 2,
        "Grassland": 1,
        "Desert": 1,
    }

    result["habitat_risk"] = (
        result["vegetation_type"]
        .map(habitat_risk_map)
        .fillna(0)
    )

    # ==================================================
    # 5. BEHAVIOR RISK
    # ==================================================

    behavior_risk_map = {
        "Running": 5,
        "Hunting": 5,
        "Alert": 4,
        "Playing": 2,
        "Socializing": 1,
        "Grazing": 1,
        "Resting": 0,
    }

    result["behavior_risk"] = (
        result["behavior"]
        .map(behavior_risk_map)
        .fillna(0)
    )

    # ==================================================
    # 6. TOTAL RISK SCORE
    # ==================================================

    result["risk_score"] = (
        result["speed_risk"]
        + result["distance_risk"]
        + result["anomaly_risk"]
        + result["weather_risk"]
        + result["habitat_risk"]
        + result["behavior_risk"]
    )

    # Cap score at 100

    result["risk_score"] = (
        result["risk_score"]
        .clip(upper=100)
        .round(2)
    )

    # ==================================================
    # 7. RISK LEVEL
    # ==================================================

    def get_risk_level(score):

        if score >= 75:
            return "Critical"

        elif score >= 50:
            return "High"

        elif score >= 25:
            return "Moderate"

        else:
            return "Low"

    result["risk_level"] = (
        result["risk_score"]
        .apply(get_risk_level)
    )

    # ==================================================
    # 8. CONSERVATION PRIORITY
    # ==================================================

    priority_map = {
        "Critical": "Immediate Attention",
        "High": "High Priority",
        "Moderate": "Monitor",
        "Low": "Normal",
    }

    result["conservation_priority"] = (
        result["risk_level"]
        .map(priority_map)
    )

    # ==================================================
    # 9. RISK REASON
    # ==================================================

    def generate_reason(row):

        reasons = []

        if row["anomaly_risk"] > 0:
            reasons.append("Movement anomaly")

        if row["speed_risk"] > 0:
            reasons.append("High movement speed")

        if row["distance_risk"] > 0:
            reasons.append("Large movement distance")

        if row["weather_risk"] > 0:
            reasons.append("Environmental condition")

        if row["habitat_risk"] > 0:
            reasons.append("Habitat factor")

        if row["behavior_risk"] > 0:
            reasons.append("Active/risk behavior")

        if not reasons:
            return "Normal movement"

        return ", ".join(reasons)

    result["risk_reason"] = result.apply(
        generate_reason,
        axis=1
    )

    return result


# ======================================================
# SPECIES RISK SUMMARY
# ======================================================

def species_risk_summary(risk_df):

    summary = (
        risk_df
        .groupby("species")
        .agg(
            average_risk_score=(
                "risk_score",
                "mean"
            ),
            maximum_risk_score=(
                "risk_score",
                "max"
            ),
            high_risk_observations=(
                "risk_score",
                lambda x: (x >= 50).sum()
            ),
            critical_observations=(
                "risk_score",
                lambda x: (x >= 75).sum()
            ),
            observations=(
                "species",
                "count"
            )
        )
        .sort_values(
            "average_risk_score",
            ascending=False
        )
        .reset_index()
    )

    return summary


# ======================================================
# CONSERVATION PRIORITY
# ======================================================

def conservation_priority(risk_df):

    priority = (
        risk_df[
            risk_df["risk_score"] >= 50
        ]
        .copy()
        .sort_values(
            "risk_score",
            ascending=False
        )
    )

    return priority


# ======================================================
# RISK DISTRIBUTION
# ======================================================

def risk_distribution(risk_df):

    return (
        risk_df["risk_level"]
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