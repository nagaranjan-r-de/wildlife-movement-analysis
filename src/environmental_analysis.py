import pandas as pd


def analyze_weather(df):
    return (
        df.groupby("weather")
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


def analyze_temperature(df):
    return (
        df.groupby("species")
        .agg(
            average_temperature=("temperature", "mean"),
            average_speed_kmh=("speed_kmh", "mean"),
            total_distance_km=("distance_km", "sum")
        )
        .sort_values(
            "average_speed_kmh",
            ascending=False
        )
    )


def analyze_vegetation(df):
    return (
        df.groupby("vegetation_type")
        .agg(
            average_speed_kmh=("speed_kmh", "mean"),
            total_distance_km=("distance_km", "sum"),
            observations=("animal_id", "count")
        )
        .sort_values(
            "total_distance_km",
            ascending=False
        )
    )


def analyze_time_of_day(df):
    return (
        df.groupby("time_of_day")
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