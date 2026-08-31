from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

OUTPUTS_DIR = BASE_DIR / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Wildlife Conservation Intelligence API",
    description="Backend API for Wildlife Movement Analysis Dashboard",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://wild-movement-analysis.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HELPER
# ============================================================

def load_csv(filename: str) -> pd.DataFrame:
    """Load an analysis CSV file."""

    file_path = ANALYSIS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Wildlife Conservation Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "analysis_directory": str(ANALYSIS_DIR),
    }


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/api/dashboard/summary")
def dashboard_summary():

    movement_df = load_csv("movement_data.csv")

    animal_df = load_csv("animal_movement_summary.csv")

    species_df = load_csv("species_movement_summary.csv")

    risk_df = load_csv("wildlife_risk_analysis.csv")

    anomalies_df = load_csv("movement_anomalies.csv")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    total_observations = len(movement_df)

    total_animals = movement_df["animal_id"].nunique()

    total_species = movement_df["species"].nunique()

    # --------------------------------------------------------
    # Movement statistics
    # --------------------------------------------------------

    total_distance = movement_df["distance_km"].sum()

    average_speed = movement_df["speed_kmh"].mean()

    # --------------------------------------------------------
    # Risk statistics
    # --------------------------------------------------------

    average_risk = risk_df["risk_score"].mean()

    maximum_risk = risk_df["risk_score"].max()

    high_risk = len(
        risk_df[risk_df["risk_level"] == "High"]
    )

    critical_risk = len(
        risk_df[risk_df["risk_level"] == "Critical"]
    )

    # --------------------------------------------------------
    # Anomaly statistics
    # --------------------------------------------------------

    movement_anomalies = len(anomalies_df)

    anomaly_percentage = (
        movement_anomalies / total_observations * 100
        if total_observations > 0
        else 0
    )

    # --------------------------------------------------------
    # Conservation status
    # --------------------------------------------------------

    if critical_risk > 0:
        conservation_status = "Critical Risk"

    elif high_risk > 0:
        conservation_status = "High Risk"

    else:
        conservation_status = "Low Risk"

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "total_observations": int(total_observations),
        "total_animals": int(total_animals),
        "total_species": int(total_species),

        "total_distance_km": round(
            float(total_distance), 4
        ),

        "average_speed_kmh": round(
            float(average_speed), 4
        ),

        "average_risk_score": round(
            float(average_risk), 3
        ),

        "maximum_risk_score": int(
            maximum_risk
        ),

        "movement_anomalies": int(
            movement_anomalies
        ),

        "anomaly_percentage": round(
            float(anomaly_percentage), 2
        ),

        "high_risk_observations": int(
            high_risk
        ),

        "critical_risk_observations": int(
            critical_risk
        ),

        "overall_conservation_status":
            conservation_status,
    }
