import pandas as pd


def top_moving_animals(df, n=10):

    result = (
        df.groupby(["animal_id", "species"])
        .agg(
            total_distance_km=("distance_km", "sum"),
            average_speed_kmh=("speed_kmh", "mean"),
            observations=("animal_id", "count")
        )
        .sort_values(
            "total_distance_km",
            ascending=False
        )
        .head(n)
    )

    return result


def behavior_movement(df):

    result = (
        df.groupby("behavior")
        .agg(
            average_speed_kmh=("speed_kmh", "mean"),
            total_distance_km=("distance_km", "sum"),
            observations=("animal_id", "count")
        )
        .sort_values(
            "average_speed_kmh",
            ascending=False
        )
    )

    return result


def habitat_preference(df):

    result = (
        df.groupby(["species", "vegetation_type"])
        .agg(
            observations=("animal_id", "count"),
            average_speed_kmh=("speed_kmh", "mean"),
            total_distance_km=("distance_km", "sum")
        )
        .sort_values(
            ["species", "observations"],
            ascending=[True, False]
        )
    )

    return result


def movement_hotspots(df):

    result = (
        df.groupby(
            ["latitude", "longitude"]
        )
        .agg(
            observations=("animal_id", "count"),
            average_speed_kmh=("speed_kmh", "mean")
        )
        .sort_values(
            "observations",
            ascending=False
        )
    )

    return result