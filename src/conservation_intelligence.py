import pandas as pd
import numpy as np


# ======================================================
# P11 — CONSERVATION INTELLIGENCE
# ======================================================

def generate_conservation_intelligence(
    df,
    risk_data,
    species_risk,
    conservation_priority,
    anomalies
):
    """
    Generate a combined conservation intelligence summary.

    Combines:
    - Wildlife movement
    - Risk analysis
    - Anomaly detection
    - Conservation priorities
    """

    print("\n" + "=" * 60)
    print("P11 - CONSERVATION INTELLIGENCE")
    print("=" * 60)

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    total_observations = len(df)

    total_animals = (
        df["animal_id"].nunique()
        if "animal_id" in df.columns
        else 0
    )

    total_species = (
        df["species"].nunique()
        if "species" in df.columns
        else 0
    )

    # --------------------------------------------------
    # Movement statistics
    # --------------------------------------------------

    average_speed = (
        df["speed_kmh"].mean()
        if "speed_kmh" in df.columns
        else 0
    )

    total_distance = (
        df["distance_km"].sum()
        if "distance_km" in df.columns
        else 0
    )

    # --------------------------------------------------
    # Risk statistics
    # --------------------------------------------------

    if risk_data is not None and not risk_data.empty:

        risk_counts = (
            risk_data["risk_level"]
            .value_counts()
        )

        low_risk = risk_counts.get("Low", 0)
        moderate_risk = risk_counts.get("Moderate", 0)
        high_risk = risk_counts.get("High", 0)
        critical_risk = risk_counts.get("Critical", 0)

        average_risk_score = (
            risk_data["risk_score"].mean()
            if "risk_score" in risk_data.columns
            else 0
        )

        maximum_risk_score = (
            risk_data["risk_score"].max()
            if "risk_score" in risk_data.columns
            else 0
        )

    else:

        low_risk = 0
        moderate_risk = 0
        high_risk = 0
        critical_risk = 0
        average_risk_score = 0
        maximum_risk_score = 0

    # --------------------------------------------------
    # Anomaly statistics
    # --------------------------------------------------

    anomaly_count = (
        len(anomalies)
        if anomalies is not None
        else 0
    )

    anomaly_percentage = (
        (anomaly_count / total_observations) * 100
        if total_observations > 0
        else 0
    )

    # --------------------------------------------------
    # Conservation priority
    # --------------------------------------------------

    high_priority_count = 0
    critical_priority_count = 0

    if (
        conservation_priority is not None
        and not conservation_priority.empty
        and "conservation_priority"
        in conservation_priority.columns
    ):

        high_priority_count = (
            conservation_priority[
                conservation_priority[
                    "conservation_priority"
                ] == "High Priority"
            ].shape[0]
        )

        critical_priority_count = (
            conservation_priority[
                conservation_priority[
                    "conservation_priority"
                ] == "Critical Priority"
            ].shape[0]
        )

    # --------------------------------------------------
    # Overall conservation status
    # --------------------------------------------------

    if critical_risk > 0:
        overall_status = "Critical"

    elif high_risk > 0:
        overall_status = "High Risk"

    elif moderate_risk > 0:
        overall_status = "Moderate Risk"

    else:
        overall_status = "Low Risk"

    # --------------------------------------------------
    # Create summary dataframe
    # --------------------------------------------------

    intelligence = pd.DataFrame({
        "metric": [
            "Total Observations",
            "Total Animals",
            "Total Species",
            "Average Speed (km/h)",
            "Total Distance (km)",
            "Average Risk Score",
            "Maximum Risk Score",
            "Low Risk Observations",
            "Moderate Risk Observations",
            "High Risk Observations",
            "Critical Risk Observations",
            "Movement Anomalies",
            "Anomaly Percentage",
            "High Conservation Priority",
            "Critical Conservation Priority",
            "Overall Conservation Status",
        ],

        "value": [
            total_observations,
            total_animals,
            total_species,
            round(average_speed, 4),
            round(total_distance, 4),
            round(average_risk_score, 4),
            maximum_risk_score,
            low_risk,
            moderate_risk,
            high_risk,
            critical_risk,
            anomaly_count,
            round(anomaly_percentage, 2),
            high_priority_count,
            critical_priority_count,
            overall_status,
        ]
    })

    # --------------------------------------------------
    # Print intelligence summary
    # --------------------------------------------------

    print("\nConservation Intelligence Summary:")
    print(intelligence.to_string(index=False))

    print("\nOverall Conservation Status:")
    print(overall_status)

    return intelligence


# ======================================================
# SPECIES CONSERVATION RANKING
# ======================================================

def species_conservation_ranking(
    species_risk
):
    """
    Rank species according to conservation risk.
    """

    if species_risk is None or species_risk.empty:
        return pd.DataFrame()

    ranking = species_risk.copy()

    if "average_risk_score" in ranking.columns:

        ranking = ranking.sort_values(
            "average_risk_score",
            ascending=False
        )

        ranking.insert(
            0,
            "conservation_rank",
            range(1, len(ranking) + 1)
        )

    return ranking


# ======================================================
# RISK DISTRIBUTION
# ======================================================

def create_risk_distribution(risk_data):
    """
    Create risk-level distribution dataframe.
    """

    if risk_data is None or risk_data.empty:
        return pd.DataFrame(
            columns=[
                "risk_level",
                "observations",
                "percentage"
            ]
        )

    distribution = (
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

    distribution.columns = [
        "risk_level",
        "observations"
    ]

    total = distribution["observations"].sum()

    distribution["percentage"] = (
        distribution["observations"] / total * 100
        if total > 0
        else 0
    )

    distribution["percentage"] = (
        distribution["percentage"]
        .round(2)
    )

    return distribution


# ======================================================
# SPECIES RISK HOTSPOTS
# ======================================================

def species_risk_hotspots(
    risk_data,
    top_n=10
):
    """
    Identify species with highest risk.
    """

    if risk_data is None or risk_data.empty:
        return pd.DataFrame()

    required_columns = {
        "species",
        "risk_score"
    }

    if not required_columns.issubset(
        risk_data.columns
    ):
        return pd.DataFrame()

    hotspots = (
        risk_data
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
            observations=(
                "risk_score",
                "count"
            )
        )
        .sort_values(
            "average_risk_score",
            ascending=False
        )
        .head(top_n)
        .reset_index()
    )

    hotspots["average_risk_score"] = (
        hotspots["average_risk_score"]
        .round(2)
    )

    return hotspots