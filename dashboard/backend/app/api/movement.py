from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api/movement",
    tags=["Movement"],
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

ANALYSIS_DIR = BASE_DIR / "outputs" / "analysis"


# ============================================================
# HELPER
# ============================================================

def load_movement_data() -> pd.DataFrame:
    file_path = ANALYSIS_DIR / "movement_data.csv"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Movement data not found: {file_path}",
        )

    return pd.read_csv(file_path)


# ============================================================
# MOVEMENT SUMMARY
# ============================================================

@router.get("/summary")
def movement_summary():

    df = load_movement_data()

    return {
        "total_observations": int(len(df)),
        "total_distance_km": round(
            float(df["distance_km"].sum()), 4
        ),
        "average_speed_kmh": round(
            float(df["speed_kmh"].mean()), 4
        ),
        "animals": int(df["animal_id"].nunique()),
        "species": int(df["species"].nunique()),
    }


# ============================================================
# MOVEMENT DATA
# ============================================================

@router.get("/data")
def movement_data():

    df = load_movement_data()

    # Convert NaN values to None-compatible values
    df = df.where(pd.notnull(df), None)

    return {
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }


# ============================================================
# SPECIES MOVEMENT
# ============================================================

@router.get("/species")
def species_movement():

    df = load_movement_data()

    result = (
        df.groupby("species")
        .agg(
            observations=("animal_id", "count"),
            total_distance_km=("distance_km", "sum"),
            average_speed_kmh=("speed_kmh", "mean"),
        )
        .reset_index()
    )

    result["total_distance_km"] = result[
        "total_distance_km"
    ].round(4)

    result["average_speed_kmh"] = result[
        "average_speed_kmh"
    ].round(4)

    return {
        "count": len(result),
        "data": result.to_dict(orient="records"),
    }


# ============================================================
# TOP MOVING ANIMALS
# ============================================================

@router.get("/top-animals")
def top_moving_animals():

    df = load_movement_data()

    result = (
        df.groupby(["animal_id", "species"])
        .agg(
            total_distance_km=("distance_km", "sum"),
            average_speed_kmh=("speed_kmh", "mean"),
            observations=("animal_id", "count"),
        )
        .reset_index()
        .sort_values(
            "total_distance_km",
            ascending=False,
        )
        .head(10)
    )

    result["total_distance_km"] = result[
        "total_distance_km"
    ].round(4)

    result["average_speed_kmh"] = result[
        "average_speed_kmh"
    ].round(4)

    return {
        "count": len(result),
        "data": result.to_dict(orient="records"),
    }
