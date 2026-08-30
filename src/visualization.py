import matplotlib.pyplot as plt
import os


OUTPUT_DIR = "outputs/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def species_distribution(df):

    plt.figure(figsize=(10, 6))

    df["species"].value_counts().plot(kind="bar")

    plt.title("Wildlife Observations by Species")
    plt.xlabel("Species")
    plt.ylabel("Number of Observations")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/species_distribution.png"
    )

    plt.close()

    print("Saved: outputs/charts/species_distribution.png")


def behavior_distribution(df):

    plt.figure(figsize=(10, 6))

    df["behavior"].value_counts().plot(kind="bar")

    plt.title("Wildlife Behavior Distribution")
    plt.xlabel("Behavior")
    plt.ylabel("Number of Observations")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/behavior_distribution.png"
    )

    plt.close()

    print("Saved: outputs/charts/behavior_distribution.png")


def location_map(df):

    plt.figure(figsize=(10, 7))

    for species in df["species"].unique():

        data = df[df["species"] == species]

        plt.scatter(
            data["longitude"],
            data["latitude"],
            label=species,
            alpha=0.6
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        "Wildlife Observation Locations"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/wildlife_locations.png"
    )

    plt.close()

    print("Saved: outputs/charts/wildlife_locations.png")

import os
import matplotlib.pyplot as plt


def species_movement(df):
    os.makedirs("outputs/charts", exist_ok=True)

    summary = (
        df.groupby("species")["distance_km"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    summary.plot(kind="bar")

    plt.title("Total Wildlife Movement by Species")
    plt.xlabel("Species")
    plt.ylabel("Total Distance (km)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        "outputs/charts/species_movement.png"
    )

    plt.close()

    print(
        "Saved: outputs/charts/species_movement.png"
    )


def weather_movement(df):
    os.makedirs("outputs/charts", exist_ok=True)

    summary = (
        df.groupby("weather")["speed_kmh"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    summary.plot(kind="bar")

    plt.title("Average Movement Speed by Weather")
    plt.xlabel("Weather")
    plt.ylabel("Average Speed (km/h)")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "outputs/charts/weather_movement.png"
    )

    plt.close()

    print(
        "Saved: outputs/charts/weather_movement.png"
    )


def vegetation_movement(df):
    os.makedirs("outputs/charts", exist_ok=True)

    summary = (
        df.groupby("vegetation_type")["speed_kmh"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    summary.plot(kind="bar")

    plt.title(
        "Average Movement Speed by Vegetation Type"
    )

    plt.xlabel("Vegetation Type")
    plt.ylabel("Average Speed (km/h)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        "outputs/charts/vegetation_movement.png"
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/vegetation_movement.png"
    )


def time_of_day_movement(df):
    os.makedirs("outputs/charts", exist_ok=True)

    summary = (
        df.groupby("time_of_day")["speed_kmh"]
        .mean()
    )

    # Keep a logical order
    order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    summary = summary.reindex(
        [x for x in order if x in summary.index]
    )

    plt.figure(figsize=(8, 5))

    summary.plot(kind="bar")

    plt.title(
        "Average Movement Speed by Time of Day"
    )

    plt.xlabel("Time of Day")
    plt.ylabel("Average Speed (km/h)")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "outputs/charts/time_of_day_movement.png"
    )

    plt.close()

    print(
        "Saved: "
        "outputs/charts/time_of_day_movement.png"
    )