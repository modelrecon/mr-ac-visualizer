import json
import pandas as pd

def load_activity_cube(path):
    with open(path, "r") as f:
        return json.load(f)


def build_heatmap_df(cube, metric="energy"):
    """
    Build a DataFrame with:
    index = layer index
    columns = tokens
    values = metric
    """
    layers = cube["layers"]
    tokens = cube["tokens"]

    data = {}
    for layer in layers:
        layer_idx = layer["layer_index"]
        if metric not in layer["core_metrics"]:
            continue
        values = layer["core_metrics"][metric]
        data[f"layer_{layer_idx}"] = values

    return pd.DataFrame(data, index=tokens).T  # tokens as columns
