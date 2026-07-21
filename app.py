import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_FILE = "pitch_value_model.pkl"
BATTER_FEAT_FILE = "batter_features.csv"
ARSENAL_FILE = "pitcher_arsenal.csv"
LOOKUP_FILE = "player_lookup.csv"

PITCH_NAMES = {
    "FF": "Four-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "FA": "Fastball (unspecified)",
    "PO": "Pitchout",
    "UN": "Unknown",
}

def pitch_display_name(code):
    return PITCH_NAMES.get(code, code)

ZONE_LEFT, ZONE_RIGHT = -0.83, 0.83
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5

def plot_strike_zone(top_row):
    """Draws the strike zone with a marker for each pitch type's best
    recommended location. The top overall recommendation is highlighted."""
    fig, ax = plt.subplots(figsize=(4.5, 5))
 
    # strike zone box
    zone = patches.Rectangle(
        (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
        linewidth=2, edgecolor="black", facecolor="none", zorder=2,
    )
    ax.add_patch(zone)
 
    # light grid lines splitting the zone into a 3x3 grid (classic
    # 9-quadrant view), just as a visual reference
    for frac in [1 / 3, 2 / 3]:
        x = ZONE_LEFT + frac * (ZONE_RIGHT - ZONE_LEFT)
        ax.plot([x, x], [ZONE_BOTTOM, ZONE_TOP], color="gray", linewidth=0.7, zorder=1)
        z = ZONE_BOTTOM + frac * (ZONE_TOP - ZONE_BOTTOM)
        ax.plot([ZONE_LEFT, ZONE_RIGHT], [z, z], color="gray", linewidth=0.7, zorder=1)
 
    ax.scatter(
        top_row["best_plate_x"], top_row["best_plate_z"],
        s=280, color="crimson", edgecolor="black", linewidth=2, zorder=4,
        label=pitch_display_name(top_row["pitch_type"]),
    )
 
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.8, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1, fontsize=8, frameon=False)
    ax.set_title("Recommended location, catcher's-eye view", fontsize=10)
    fig.tight_layout()
    return fig

BATTER_FEATURES = ["avg_delta_run_exp", "rv_fastball", "rv_breaking", "rv_offspeed",
                   "xwoba_fastball", "xwoba_breaking", "xwoba_offspeed",
                   "avg_velo_faced", "corr_velo_rv", "swing_rate", "whiff_rate",
                   "chase_rate", "vs_rhp_rv", "vs_lhp_rv"]

PLATE_X_GRID = np.arange(-1.5, 1.51, 0.25)
PLATE_Z_GRID = np.arange(1.0, 4.01, 0.25)

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]

@st.cache_data
def load_data():
    batter_feat = pd.read_csv(BATTER_FEAT_FILE)
    arsenal = pd.read_csv(ARSENAL_FILE)
    lookup = pd.read_csv(LOOKUP_FILE)
    return batter_feat, arsenal, lookup

def name_options(lookup, ids_present):
    sub = lookup[lookup["mlbam_id"].isin(ids_present)].copy()
    sub = sub.dropna(subset=["full_name"]).sort_values("full_name")
    return sub

def score_pitch_type(model, feature_cols, pitcher_row, batter_row, balls, strikes, batter_stand, pitcher_throws):
    n_x, n_z = len(PLATE_X_GRID), len(PLATE_Z_GRID)
    grid = pd.DataFrame({
        "plate_x": np.repeat(PLATE_X_GRID, n_z),
        "plate_z": np.tile(PLATE_Z_GRID, n_x),
    })
    grid["release_speed"] = pitcher_row["velo_mean"]
    grid["release_spin_rate"] = pitcher_row["spin_mean"]
    grid["pfx_x"] = pitcher_row["hb_mean"] / 12.0
    grid["pfx_z"] = pitcher_row["ivb_mean"] / 12.0
    grid["release_extension"] = pitcher_row["extension_mean"]
    grid["balls"] = balls
    grid["strikes"] = strikes

    for bf in BATTER_FEATURES:
        grid[bf] = batter_row.get(bf, np.nan)

    for c in feature_cols:
        if c.startswith("pitch_type_"):
            grid[c] = 1 if c == f"pitch_type_{pitcher_row['pitch_type']}" else 0
        elif c.startswith("stand_"):
            grid[c] = 1 if c == f"stand_{batter_stand}" else 0
        elif c.startswith("p_throws_"):
            grid[c] = 1 if c == f"p_throws_{pitcher_throws}" else 0
    
    X = grid.reindex(columns = feature_cols, fill_value = 0)
    grid["predicted_rv"] = model.predict(X)
    best = grid.loc[grid["predicted_rv"].idxmin()]
    return best, grid

def main():
    model, feature_cols = load_model()
    batter_feat, arsenal, lookup = load_data()

    pitcher_ids = arsenal["pitcher"].unique()
    batter_ids = batter_feat["batter"].unique()

    pitcher_names = name_options(lookup, pitcher_ids)
    batter_names = name_options(lookup, batter_ids)

    col1, col2 = st.columns(2)
    with col1:
        pitcher_name = st.selectbox("Pitcher", pitcher_names["full_name"])
        pitcher_id = pitcher_names.loc[pitcher_names["full_name"] == pitcher_name, "mlbam_id"].iloc[0]
    with col2:
        batter_name = st.selectbox("Batter", batter_names["full_name"])
        batter_id = batter_names.loc[batter_names["full_name"] == batter_name, "mlbam_id"].iloc[0]
        _stand_preview = batter_feat.loc[batter_feat["batter"] == batter_id, "stand"]
        if not _stand_preview.empty:
            st.caption(f"Bats: {_stand_preview.iloc[0]}")
    
    col3, col4 = st.columns(2)
    with col3:
        balls = st.selectbox("Balls", [0, 1, 2, 3], index = 0)
    with col4:
        strikes = st.selectbox("Strikes", [0, 1, 2], index = 0)
    
    if st.button("Get Recommendation", type = "primary"):
        pitcher_arsenal = arsenal[arsenal["pitcher"] == pitcher_id].sort_values("usage_rate", ascending = False)
        batter_row = batter_feat[batter_feat["batter"] == batter_id].iloc[0]

        if pitcher_arsenal.empty:
            st.error("No arsenal data found for pitcher")
            return
        
        batter_stand = batter_row["stand"]
        pitcher_throws = pitcher_arsenal["p_throws"].iloc[0]

        results = []
        for _, prow in pitcher_arsenal.iterrows():
            best, _ = score_pitch_type(
                model, feature_cols, prow, batter_row, balls, strikes, batter_stand, pitcher_throws
            )
            results.append({
                "pitch_type": prow["pitch_type"],
                "usage_rate": prow["usage_rate"],
                "avg_velo": prow["velo_mean"],
                "best_plate_x": best["plate_x"],
                "best_plate_z": best["plate_z"],
                "predicted_run_value": best["predicted_rv"],
            })

        results_df = pd.DataFrame(results).sort_values("predicted_run_value").reset_index(drop=True)
        results_df["pitch_name"] = results_df["pitch_type"].apply(pitch_display_name)

        top = results_df.iloc[0]
        st.subheader(f"Recommendation: {top['pitch_name']}")
        plot_col, info_col = st.columns([1, 1])
        with plot_col:
            fig = plot_strike_zone(top)
            st.pyplot(fig)
        with info_col:
            st.metric("Predicted run value", f"{top['predicted_run_value']:+.4f}")
            st.caption(f"Matchup: {pitcher_throws}HP vs. {batter_stand}HB")

        st.subheader("All Pitch Types, Ranked")
        st.dataframe(
            results_df.style.format({
                "usage_rate": "{:.1%}",
                "avg_velo": "{:.1f}",
                "best_plate_x": "{:.2f}",
                "best_plate_z": "{:.2f}",
                "predicted_run_value": "{:+.4f}",
            }),
            use_container_width = True,
        )

if __name__ == "__main__":
    main()