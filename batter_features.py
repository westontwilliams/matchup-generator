import pandas as pd
import numpy as np

INFILE = "statcast.parquet"
OUTFILE = "batter_features.csv"

BREAKING = {"SL", "CU", "KC", "ST", "SV", "CS"}
FASTBALL = {"FF", "SI", "FC"}
OFFSPEED = {"CH", "FS", "FO", "SC"}

def pitch_family(pt):
    if pt in BREAKING:
        return "breaking"
    elif pt in FASTBALL:
        return "fastball"
    elif pt in OFFSPEED:
        return "offspeed"
    return "other"

SWING_DESC = {"hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
              "foul_tip", "foul_bunt", "missed_bunt"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}

CHASE_ZONES = set(range(11, 15))

def main():
    df = pd.read_parquet(INFILE)
    df = df.dropna(subset=["batter"])

    df["pitch_family"] = df["pitch_type"].apply(pitch_family)
    df["is_swing"] = df["description"].isin(SWING_DESC)
    df["is_whiff"] = df["description"].isin(WHIFF_DESC)
    df["is_chase"] = df["is_swing"] & df["zone"].isin(CHASE_ZONES)
    df["is_out_of_zone"] = df["zone"].isin(CHASE_ZONES)

    rows = []
    for batter_id, g in df.groupby("batter"):
        n_pitches = len(g)
        if n_pitches < 100:
            continue
        feat = {"batter": batter_id, "n_pitches_season": n_pitches}

        feat["stand"] = g["stand"].mode().iloc[0] if "stand" in g.columns else np.nan

        feat["avg_delta_run_exp"] = g["delta_run_exp"].mean()

        for fam in ["fastball", "breaking", "offspeed"]:
            sub = g[g["pitch_family"] == fam]
            feat[f"n_{fam}"] = len(sub)
            feat[f"rv_{fam}"] = sub["delta_run_exp"].mean() if len(sub) >= 20 else np.nan

            feat[f"xwoba_{fam}"] = sub["estimated_woba_using_speedangle"].mean() if sub["estimated_woba_using_speedangle"].notna().sum() >= 15 else np.nan

        vel_valid = g.dropna(subset = ["release_speed", "delta_run_exp"])
        feat["avg_velo_faced"] = vel_valid["release_speed"].mean()
        if len(vel_valid) >= 50:
            feat["corr_velo_rv"] = vel_valid["release_speed"].corr(vel_valid["delta_run_exp"])
        else:
            feat["corr_velo_rv"] = np.nan

        feat["swing_rate"] = g["is_swing"].mean()
        feat["whiff_rate"] = g.loc[g["is_swing"], "is_whiff"].mean() if g["is_swing"].sum() > 0 else np.nan
        feat["chase_rate"] = g.loc[g["is_out_of_zone"], "is_swing"].mean() if g["is_out_of_zone"].sum() > 0 else np.nan

        for throws, label in [("R", "vs_rhp"), ("L", "vs_lhp")]:
            sub = g[g["p_throws"] == throws]
            feat[f"{label}_rv"] = sub["delta_run_exp"].mean() if len(sub) >= 50 else np.nan
            feat[f"{label}_n"] = len(sub)

        rows.append(feat)

    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index = False)
    print(out.head(5).to_string(index = False))

if __name__ == "__main__":
    main()