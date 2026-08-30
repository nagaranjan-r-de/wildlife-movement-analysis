import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates in kilometers.
    """

    earth_radius = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c


def calculate_movement(df):
    df = df.copy()

    # Sort observations by animal and time
    df = df.sort_values(
        ["animal_id", "timestamp"]
    ).reset_index(drop=True)

    # Previous GPS location
    df["previous_latitude"] = (
        df.groupby("animal_id")["latitude"].shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("animal_id")["longitude"].shift(1)
    )

    # Previous timestamp
    df["previous_timestamp"] = (
        df.groupby("animal_id")["timestamp"].shift(1)
    )

    # Distance
    df["distance_km"] = haversine(
        df["previous_latitude"],
        df["previous_longitude"],
        df["latitude"],
        df["longitude"]
    )

    # Time difference
    df["time_diff_hours"] = (
        (df["timestamp"] - df["previous_timestamp"])
        .dt.total_seconds()
        / 3600
    )

    # Speed
    df["speed_kmh"] = (
        df["distance_km"] /
        df["time_diff_hours"]
    )

    return df