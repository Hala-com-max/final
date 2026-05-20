import pandas as pd
from a4s_eval.data_model.measure import Measure
import os

OUTPUT_FOLDER = "./tests/data/measures/"

def save_measures(name: str, measures: list[Measure]) -> None:
    df = pd.DataFrame([m.model_dump() for m in measures])
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_path = OUTPUT_FOLDER + name.lower().replace(" ", "_") + ".csv"

    # Write header only if file does NOT exist
    write_header = not os.path.exists(file_path)

    df.to_csv(
        file_path,
        mode="a",          # append mode
        index=False,
        header=write_header
    )

    print(f"Saved {len(measures)} measures → {file_path}")
