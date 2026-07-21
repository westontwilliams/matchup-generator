import pandas as pd
import numpy as np

from pybaseball import playerid_reverse_lookup

PITCH_FILE = "statcast.parquet"
OUTFILE = "pitcher_arsenal.csv"
LOOKUP_OUTFILE = "player_lookup.csv"

MIN_PITCHES = 15

def main():
    df = pd.read_parquet(PITCH_FILE)
    df = df.dropna(subset = ["pitcher", "pitch_type", "release_speed"])

    rows = []
    for pitcher_id, g in df.groupby("pitcher"):
        total_pitches = len(g)
        for pitch_type, gp in g.groupby("pitch_type"):
            n = len(gp)
            if n < MIN_PITCHES:
                continue
            rows.append({
                "pitcher": pitcher_id,
                "pitch_type": pitch_type,
                "n_pitches": n,
                "usage_rate": n/total_pitches,
                "velo_mean": gp["release_speed"].mean(),
                "velo_std": gp["release_speed"].std(),
                "spin_mean": gp["release_spin_rate"].mean(),
                "spin_std": gp["release_spin_rate"].std(),
                "ivb_mean": (gp["pfx_z"] * 12).mean(),
                "ivb_std": (gp["pfx_z"] * 12).std(),
                "hb_mean": (gp["pfx_x"] * 12).mean(),
                "hb_std": (gp["pfx_x"] * 12).std(),
                "extension_mean": gp["release_extension"].mean() if "release_extension" in gp.columns else np.nan,
                "p_throws": gp["p_throws"].iloc[0] if "p_throws" in gp.columns else np.nan,
            })
    
    arsenal = pd.DataFrame(rows)
    arsenal.to_csv(OUTFILE, index = False)

    all_ids = pd.concat([df["pitcher"], df["batter"]]).dropna().astype(int).unique().tolist()
    lookup = playerid_reverse_lookup(all_ids, key_type = "mlbam")
    lookup = lookup[["key_mlbam", "name_first", "name_last"]].copy()
    lookup["full_name"] = lookup["name_first"].str.title() + " " + lookup["name_last"].str.title()
    lookup = lookup[["key_mlbam", "full_name"]].rename(columns = {"key_mlbam": "mlbam_id"})
    lookup.to_csv(LOOKUP_OUTFILE, index = False)

if __name__ == "__main__":
    main()