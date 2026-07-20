import pandas as pd
import numpy as np

INFILE = "statcast.parquet"

def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["velo_band"] = pd.cut(
        df["release_speed"],
        bins = np.arange(60, 105, 3),
        include_lowest = True
    )

    df["ivb_in"] = df["pfx_z"] * 12
    df["hb_in"] = df["pfx_x"] * 12

    df["ivb_band"] = pd.cut(
        df["ivb_in"],
        bins = np.arange(-24, 26, 4),
        include_lowest = True
    )

    df["hb_band"] = pd.cut(
        df["hb_in"],
        bins = np.arange(-24, 26, 4),
        include_lowest = True
    )

    def count_group(row):
        b, s = row["balls"], row["strikes"]
        return f"{b}-{s}"
    
    df["count_group"] = df.apply(count_group, axis = 1)

    return df


def main():
    df = pd.read_parquet(INFILE)
    df = df.dropna(subset = ["release_speed", "pfx_x", "pfx_z", "pitch_type"])
    df = add_buckets(df)

    bucket_cols_coarse = ["batter", "pitch_type", "velo_band", "count_group"]

    coarse = df.groupby(bucket_cols_coarse, observed = True).size().reset_index(name = "n_pitches")

    print("\n=== COARSE BUCKETS (pitch type x velo band x count group), per batter ===")
    print(f"Total buckets created: {len(coarse):,}")
    print(coarse["n_pitches"].describe())
    print("\nShare of buckets with fewer than 10 pitches:",
          f"{(coarse['n_pitches'] < 10).mean():.1%}")
    print("Share of buckets with fewer than 5 pitches:",
          f"{(coarse['n_pitches'] < 5).mean():.1%}")
    print("Share of buckets with >= 20 pitches (reasonably trustworthy solo):",
          f"{(coarse['n_pitches'] >= 20).mean():.1%}")

    bucket_cols_fine = ["batter", "pitch_type", "velo_band", "ivb_band", "hb_band", "count_group"]

    fine = df.groupby(bucket_cols_fine, observed = True).size().reset_index(name = "n_pitches")

    print("\n=== FINE BUCKETS (+ IVB band + HB band), per batter ===")
    print(f"Total buckets created: {len(fine):,}")
    print(fine["n_pitches"].describe())
    print("\nShare of buckets with fewer than 10 pitches:",
          f"{(fine['n_pitches'] < 10).mean():.1%}")
    print("Share of buckets with fewer than 5 pitches:",
          f"{(fine['n_pitches'] < 5).mean():.1%}")
    print("Share of buckets with >= 20 pitches:",
          f"{(fine['n_pitches'] >= 20).mean():.1%}")
    
    example_batter = df["batter"].value_counts().index[0]  # most-seen batter in dataset
    ex = coarse[coarse["batter"] == example_batter].sort_values("n_pitches", ascending=False)
    print(f"\n=== Example: batter id {example_batter} (most total pitches in dataset) ===")
    print("Top 10 coarse buckets by pitch count:")
    print(ex.head(10).to_string(index=False))
    print(f"\nTotal coarse buckets for this batter: {len(ex)}, "
          f"median bucket size: {ex['n_pitches'].median():.1f}")

    coarse.to_csv("coarse_buckets.csv", index = False)
    fine.to_csv("fine_buckets.csv", index = False)

if __name__ == "__main__":
    main()