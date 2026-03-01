import pandas as pd
import json

WINDOW_SIZE = 30

df = pd.read_csv("data/processed/train_final.csv")

# Select first engine for sample
engine_id = df["engine_id"].unique()[0]
engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

features = engine_df.drop(
    columns=["engine_id", "cycle", "RUL", "failure"]
).values

total_cycles = len(features)

# Middle lifecycle window
mid_index = total_cycles // 2
start_index = max(0, mid_index - WINDOW_SIZE)

sample_sequence = features[start_index:mid_index]

output = {
    "engine_id": int(engine_id),
    "window_type": "middle_lifecycle",
    "sequence_shape": list(sample_sequence.shape),
    "sequence": sample_sequence.tolist()
}

print(json.dumps(output, indent=2))