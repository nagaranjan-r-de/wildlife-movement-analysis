import pandas as pd


def create_trajectories(df, animals_per_species=10):

    df = df.copy()

    # Sort observations by species and timestamp
    df = df.sort_values(
        ["species", "timestamp"]
    ).reset_index(drop=True)

    animal_ids = {}

    # Create animal IDs for each species
    for species in df["species"].unique():

        prefix = species[:3].upper()

        animal_ids[species] = [
            f"{prefix}_{i:03d}"
            for i in range(
                1,
                animals_per_species + 1
            )
        ]

    # Assign observations sequentially
    # instead of randomly
    df["animal_number"] = (
        df.groupby("species")
        .cumcount()
        % animals_per_species
    )

    df["animal_id"] = [
        animal_ids[species][number]
        for species, number
        in zip(
            df["species"],
            df["animal_number"]
        )
    ]

    df = df.drop(
        columns=["animal_number"]
    )

    return df
