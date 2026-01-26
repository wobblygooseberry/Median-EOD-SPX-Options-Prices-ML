import csv, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CSV = "spx_all_2010_2023.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_FEATS = [
    "UNDERLYING_LAST","STRIKE","DTE",
    "C_BID","C_ASK","P_BID","P_ASK",
    "C_DELTA","C_GAMMA","C_VEGA","C_THETA","C_RHO","C_IV",
    "STRIKE_DISTANCE","STRIKE_DISTANCE_PCT",
]
ENGINEERED = ["C_MID","P_MID","C_SPREAD","P_SPREAD","MONEYNESS","LOG_MONEYNESS","SQRT_TAU"]
ALL_FEATS = BASE_FEATS + ENGINEERED
CRITICAL  = ["UNDERLYING_LAST","STRIKE","DTE","C_BID","C_ASK"]

def normalize_headers(cols):
    cleaned = []
    for c in cols:
        c = str(c)
        c = c.replace("[","").replace("]","").replace('"',"").replace("'","")
        c = c.replace("\u00A0"," ").replace("\u2009"," ").replace("\u2002"," ").replace("\u2003"," ")
        c = re.sub(r"^\s+|\s+$", "", c)
        c = re.sub(r"\s+", "_", c)
        cleaned.append(c)
    return cleaned

def coerce_numerics(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def engineer(df):
    if {"C_BID","C_ASK"}.issubset(df.columns):
        df["C_MID"] = (df["C_BID"] + df["C_ASK"]) / 2.0
        df["C_SPREAD"] = df["C_ASK"] - df["C_BID"]
    else:
        df["C_MID"] = np.nan; df["C_SPREAD"] = np.nan
    if {"P_BID","P_ASK"}.issubset(df.columns):
        df["P_MID"] = (df["P_BID"] + df["P_ASK"]) / 2.0
        df["P_SPREAD"] = df["P_ASK"] - df["P_BID"]
    else:
        df["P_MID"] = np.nan; df["P_SPREAD"] = np.nan
    if {"UNDERLYING_LAST","STRIKE"}.issubset(df.columns):
        df["MONEYNESS"] = df["UNDERLYING_LAST"] / df["STRIKE"]
        df["LOG_MONEYNESS"] = np.log(df["MONEYNESS"])
    else:
        df["MONEYNESS"] = np.nan; df["LOG_MONEYNESS"] = np.nan
    df["SQRT_TAU"] = np.sqrt(df["DTE"]/365.0) if "DTE" in df.columns else np.nan
    return df

# 1) load stats + model
stats = np.load("mlp_norm_stats.npz")
feat_mean = stats["feat_mean"]
feat_std  = stats["feat_std"]
t_mean    = float(stats["t_mean"])
t_std     = float(stats["t_std"])

model = nn.Sequential(
    nn.Linear(len(ALL_FEATS), 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
).to(DEVICE)
state_dict = torch.load("mlp_spx_mid.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# 2) take a sample of the data (e.g. first 50k rows)
print("Loading sample data...")
df = pd.read_csv(
    CSV,
    nrows=50000,                 # change if you want more/less
    sep=",",
    engine="python",
    quoting=csv.QUOTE_NONE,
    skipinitialspace=True,
    on_bad_lines="skip",
)

df.columns = normalize_headers(df.columns)
coerce_numerics(df, list(set(CRITICAL + BASE_FEATS)))
df = engineer(df)

# require CRITICAL
df = df.dropna(subset=CRITICAL)
if df.empty:
    raise SystemExit("No valid rows after cleaning.")

# fill remaining NaNs
cols_for_model = list(set(ALL_FEATS + ["C_MID"]))
df[cols_for_model] = df[cols_for_model].fillna(0.0)

X = df[ALL_FEATS].to_numpy()
y_true = df["C_MID"].to_numpy()

mask = np.isfinite(X).all(axis=1) & np.isfinite(y_true)
X = X[mask]; y_true = y_true[mask]

# standardize and predict
X_std = (X - feat_mean) / feat_std
X_t = torch.tensor(X_std, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    pred_std = model(X_t).cpu().numpy().ravel()
y_pred = pred_std * t_std + t_mean   # back to dollars

# 3) make scatter plot
print("Making scatter plot...")
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.2, s=5, label="Model predictions")
# red 45° line = true mid-price (perfect fit)
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linewidth=2, label="Perfect fit")

plt.xlabel("Actual Call Mid-Price ($)")
plt.ylabel("Model-Predicted Mid-Price ($)")
plt.title("Neural Network Pricing: Predicted vs Actual Mid-Prices")
plt.legend()
plt.tight_layout()
plt.savefig("mid_scatter.png", dpi=300)
print("Saved mid_scatter.png")
