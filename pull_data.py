import os
import pandas as pd
from pybaseball import statcast, cache

START_DATE = "2025-03-18"
END_DATE = "2026-07-19"
OUTFILE = "statcast.parquet"

def main():
    cache.enable()

    if os.path.exists(OUTFILE):
        print(f"File {OUTFILE} already exists. Delete if you want to re-pull.")
        return

    df = statcast(start_dt=START_DATE, end_dt=END_DATE)

    keep_cols = [
        "game_date", "player_name", "pitcher", "batter", "pitch_type", "pitch_name",
        "release_speed", "release_spin_rate", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", 
        "plate_x", "plate_z", "release_extension", "release_pos_y",
        "balls", "strikes", "stand", "p_throws", "outs_when_up", "inning", "description",
        "events", "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "launch_speed", "launch_angle", "zone", "type",
        "delta_run_exp"
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df.to_parquet(OUTFILE, index = False)

if __name__ == "__main__":
     main()