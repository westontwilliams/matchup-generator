import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle

PITCH_FILE = "statcast.parquet"
BATTER_FEAT_FILE = "batter_features.csv"
MODEL_OUT = "pitch_value_model.pkl"

PITCH_LEVEL_FEATURES = ["release_speed", "release_spin_rate", "release_extension","pfx_x", "pfx_z",
                        "plate_x", "plate_z", "balls", "strikes"]

CATEGORICAL_FEATURES = ["pitch_type", "stand", "p_throws"]

BATTER_FEATURES = ["avg_delta_run_exp", "rv_fastball", "rv_breaking", "rv_offspeed",
                   "xwoba_fastball", "xwoba_breaking", "xwoba_offspeed",
                   "avg_velo_faced", "corr_velo_rv", "swing_rate", "whiff_rate",
                   "chase_rate", "vs_rhp_rv", "vs_lhp_rv"]

TARGET = "delta_run_exp"

def main():
    pitches = pd.read_parquet(PITCH_FILE)
    batter_feat = pd.read_csv(BATTER_FEAT_FILE)

    df = pitches.merge(batter_feat.drop(columns=["stand"], errors="ignore"), on="batter", how="inner")
    
    needed_cols = PITCH_LEVEL_FEATURES + CATEGORICAL_FEATURES + BATTER_FEATURES + [TARGET]

    df = df.dropna(subset = [c for c in needed_cols if c in df.columns])
    df = pd.get_dummies(df, columns = CATEGORICAL_FEATURES, drop_first = False)
    dummy_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]

    feature_cols = PITCH_LEVEL_FEATURES + dummy_cols + BATTER_FEATURES

    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = xgb.XGBRegressor(n_estimators = 400, max_depth = 6, learning_rate = 0.03, subsample = 0.8,
                             colsample_bytree = 0.8, reg_lambda = 1.0, random_state = 42, n_jobs = -1)
    
    model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    print(f"\nModel MAE: {mae:.4f}  (baseline MAE predicting the mean: {baseline_mae:.4f})")
    print(f"Model R^2: {r2:.4f}")
    print("(Per-pitch run value is inherently very noisy -- most of a pitch's")
    print(" delta_run_exp is driven by outcome randomness, not predictable")
    print(" pitch quality, so don't expect a high R^2 here. What matters is")
    print(" whether the model beats the naive baseline and whether its")
    print(" feature importances / rankings make baseball sense.)")
 
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importances.head(15).to_string())
 
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"\nSaved model to {MODEL_OUT}")
 
    print("\n=== Demo: predicted run value for one batter across a few pitch shapes ===")
    demo_batter_id = batter_feat.loc[batter_feat["n_pitches_season"].idxmax(), "batter"]
    demo_feat_row = batter_feat[batter_feat["batter"] == demo_batter_id].iloc[0]
 
    demo_pitches = [
        ("95mph FF, middle-middle, 0-0", 95, 2300,  0.0,  1.4, 0.0, 2.5, 0, 0, 6.3, "FF"),
        ("95mph FF, up-and-in, 0-0", 95, 2300,  0.5,  1.5, -0.8, 3.3, 0, 0, 6.3, "FF"),
        ("84mph SL, low-and-away, 0-2", 84, 2600, -0.9, -0.2, 0.9, 1.5, 0, 2, 6.3, "SL"),
        ("78mph CU, backdoor, 0-2", 78, 2500, -0.6,  0.3, 0.7, 2.0, 0, 2, 6.3,"CU"),
    ]
 
    demo_rows = []
    for label, velo, spin, pfx_x, pfx_z, px, pz, b, s, extension, ptype in demo_pitches:
        row = {
            "release_speed": velo, "release_spin_rate": spin,
            "pfx_x": pfx_x, "pfx_z": pfx_z,
            "plate_x": px, "plate_z": pz,
            "balls": b, "strikes": s, "release_extension": extension
        }
        for bf in BATTER_FEATURES:
            row[bf] = demo_feat_row.get(bf, np.nan)
        for c in dummy_cols:
            row[c] = 0
        pt_col = f"pitch_type_{ptype}"
        if pt_col in row:
            row[pt_col] = 1
        for c in ["stand_R", "p_throws_R"]:
            if c in row:
                row[c] = 1
        demo_rows.append((label, row))
 
    demo_X = pd.DataFrame([r for _, r in demo_rows])[feature_cols].fillna(0)
    demo_preds = model.predict(demo_X)
    for (label, _), pred in zip(demo_rows, demo_preds):
        print(f"  {label:35s} predicted delta_run_exp: {pred:+.4f}  "
              f"({'good for pitcher' if pred < 0 else 'good for batter'})")
 
 
if __name__ == "__main__":
    main()


