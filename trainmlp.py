import csv, re, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- CONFIG ----------------
CSV    = "spx_all_2010_2023.csv"
TARGET = "C_MID"                  # mid-price target
CHUNK  = 200_000                  # drop to 100_000 if memory tight
BATCH  = 2048
EPOCHS = 3
LOG_EVERY = 5                     # print every N chunks
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

# EXACT columns in your file
BASE_FEATS = [
    "UNDERLYING_LAST","STRIKE","DTE",
    "C_BID","C_ASK","P_BID","P_ASK",
    "C_DELTA","C_GAMMA","C_VEGA","C_THETA","C_RHO","C_IV",
    "STRIKE_DISTANCE","STRIKE_DISTANCE_PCT",
]
ENGINEERED = ["C_MID","P_MID","C_SPREAD","P_SPREAD","MONEYNESS","LOG_MONEYNESS","SQRT_TAU"]
ALL_FEATS = BASE_FEATS + ENGINEERED

# critical columns that MUST be present + non-null for a row to be usable
CRITICAL = ["UNDERLYING_LAST","STRIKE","DTE","C_BID","C_ASK"]

def read_chunks(path, chunksize):
    """Read CSV where whole lines may be quoted; ignore quotes so commas split correctly."""
    return pd.read_csv(
        path,
        chunksize=chunksize,
        sep=",",
        engine="python",
        quoting=csv.QUOTE_NONE,
        skipinitialspace=True,
        on_bad_lines="skip",
    )

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
    # bids/asks → mids & spreads
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

    # moneyness, log moneyness, sqrt time
    if {"UNDERLYING_LAST","STRIKE"}.issubset(df.columns):
        df["MONEYNESS"] = df["UNDERLYING_LAST"] / df["STRIKE"]
        df["LOG_MONEYNESS"] = np.log(df["MONEYNESS"])
    else:
        df["MONEYNESS"] = np.nan; df["LOG_MONEYNESS"] = np.nan

    df["SQRT_TAU"] = np.sqrt(df["DTE"]/365.0) if "DTE" in df.columns else np.nan
    return df

def batches(X, y, bs):
    for i in range(0, len(X), bs):
        yield X[i:i+bs], y[i:i+bs]

def combine_stats(n_a, mu_a, m2_a, n_b, mu_b, m2_b):
    """Combine two sets of (n, mean, m2) for vectorized features."""
    if n_b == 0: return n_a, mu_a, m2_a
    if n_a == 0: return n_b, mu_b, m2_b
    delta = mu_b - mu_a
    n_tot = n_a + n_b
    mu_tot = mu_a + delta * (n_b / n_tot)
    m2_tot = m2_a + m2_b + (delta * delta) * (n_a * n_b / n_tot)
    return n_tot, mu_tot, m2_tot

# --------------- PASS 1: fast stats + chatty ---------------
t0 = time.time()
mu = np.zeros(len(ALL_FEATS), dtype=np.float64)
m2 = np.zeros(len(ALL_FEATS), dtype=np.float64)
n_feat = 0
t_mu = 0.0; t_m2 = 0.0; t_n = 0
printed_headers = False
chunks = 0
rows_used = 0
rows_seen = 0

for chunk in read_chunks(CSV, CHUNK):
    chunk.columns = normalize_headers(chunk.columns)
    if not printed_headers:
        print("Headers after normalization:", chunk.columns.tolist()[:40])
        printed_headers = True

    # require at least CRITICAL columns to be present
    if set(CRITICAL) - set(chunk.columns):
        chunks += 1
        if chunks % LOG_EVERY == 0:
            print(f"[PASS 1] skipped chunk (missing critical cols). chunks={chunks}, rows_used={rows_used:,}, elapsed={time.time()-t0:.1f}s")
        continue

    # numeric conversion for relevant columns (critical + base)
    coerce_numerics(chunk, list(set(CRITICAL + BASE_FEATS)))

    # engineer features (this creates C_MID)
    chunk = engineer(chunk)

    before = len(chunk)

    # drop rows missing critical features only
    chunk = chunk.dropna(subset=CRITICAL)
    after = len(chunk)
    rows_seen += before
    if after == 0:
        chunks += 1
        if chunks % LOG_EVERY == 0:
            print(f"[PASS 1] dropped all rows by CRITICAL (before={before:,}). chunks={chunks}, rows_used={rows_used:,}")
        continue

    # fill remaining NaNs in ALL_FEATS + TARGET with 0 (keep as much data as possible)
    cols_for_model = list(set(ALL_FEATS + [TARGET]))
    chunk[cols_for_model] = chunk[cols_for_model].fillna(0.0)

    # build matrices
    X = chunk[ALL_FEATS].to_numpy(dtype=np.float64)
    y = chunk[TARGET].to_numpy(dtype=np.float64)

    # filter out any remaining non-finite rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]; y = y[mask]
    used = len(X)
    rows_used += used

    # vectorized stats for features
    if used > 0:
        n_b = used
        mu_b = X.mean(axis=0)
        m2_b = ((X - mu_b)**2).sum(axis=0)
        n_feat, mu, m2 = combine_stats(n_feat, mu, m2, n_b, mu_b, m2_b)

        # target stats
        n_y = used
        mu_y = y.mean()
        m2_y = ((y - mu_y)**2).sum()
        t_n, t_mu, t_m2 = combine_stats(
            t_n, np.array([t_mu]), np.array([t_m2]),
            n_y, np.array([mu_y]), np.array([m2_y])
        )
        t_mu = float(t_mu[0]); t_m2 = float(t_m2[0])

    chunks += 1
    if chunks % LOG_EVERY == 0:
        print(f"[PASS 1] chunks={chunks}, rows_used={rows_used:,}, "
              f"running_feat_n={n_feat:,}, elapsed={time.time()-t0:.1f}s")

feat_mean = mu
feat_std  = np.sqrt(m2 / max(n_feat - 1, 1)); feat_std[feat_std == 0] = 1.0
t_mean    = float(t_mu)
t_std     = float(np.sqrt(t_m2 / max(t_n - 1, 1))) if t_n > 1 else 1.0
t_std     = 1.0 if t_std == 0 else t_std

print(f"[PASS 1] DONE  rows_used={rows_used:,}  feat_n={n_feat:,}  target_n={t_n:,}  took={time.time()-t0:.1f}s")

# --------------- MODEL ---------------
model = nn.Sequential(
    nn.Linear(len(ALL_FEATS), 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --------------- PASS 2: train + chatty ---------------
for epoch in range(EPOCHS):
    t1 = time.time()
    total_loss = 0.0
    total_count = 0
    chunks = 0
    rows_used = 0

    for chunk in read_chunks(CSV, CHUNK):
        chunk.columns = normalize_headers(chunk.columns)

        # need at least CRITICAL columns
        if set(CRITICAL) - set(chunk.columns):
            continue

        coerce_numerics(chunk, list(set(CRITICAL + BASE_FEATS)))
        chunk = engineer(chunk)

        # drop rows missing CRITICAL only
        chunk = chunk.dropna(subset=CRITICAL)
        if chunk.empty:
            continue

        # fill remaining NaNs in ALL_FEATS + TARGET
        cols_for_model = list(set(ALL_FEATS + [TARGET]))
        chunk[cols_for_model] = chunk[cols_for_model].fillna(0.0)

        X = chunk[ALL_FEATS].to_numpy()
        y = chunk[TARGET].to_numpy().reshape(-1,1)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).ravel()
        if not mask.any():
            continue
        X = X[mask]; y = y[mask]
        rows_used += len(X)

        # standardize with PASS 1 stats
        X = (X - feat_mean) / feat_std
        y = (y - t_mean) / t_std

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        model.train()
        for xb, yb in batches(X, y, BATCH):
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            total_count += len(xb)

        chunks += 1
        if chunks % LOG_EVERY == 0:
            print(f"[E{epoch+1}] chunks={chunks}, rows_used={rows_used:,}, "
                  f"running_mse={total_loss/max(total_count,1):.6f}")

    print(f"Epoch {epoch+1}/{EPOCHS}  train_mse={total_loss/max(total_count,1):.6f}  "
          f"rows_used={rows_used:,}  took={time.time()-t1:.1f}s")

print("Training done. Now running evaluation...")

# --------------- PASS 3: evaluation (RMSE, MAE, R² only) ---------------

model.eval()
eval_n = 0
sse = 0.0       # sum of squared errors
sae = 0.0       # sum of absolute errors
sum_y = 0.0     # sum of true y
sum_y2 = 0.0    # sum of y^2

with torch.no_grad():
    for chunk in read_chunks(CSV, CHUNK):
        chunk.columns = normalize_headers(chunk.columns)

        if set(CRITICAL) - set(chunk.columns):
            continue

        coerce_numerics(chunk, list(set(CRITICAL + BASE_FEATS)))
        chunk = engineer(chunk)

        # drop rows missing CRITICAL only
        chunk = chunk.dropna(subset=CRITICAL)
        if chunk.empty:
            continue

        # fill remaining NaNs in ALL_FEATS + TARGET
        cols_for_model = list(set(ALL_FEATS + [TARGET]))
        chunk[cols_for_model] = chunk[cols_for_model].fillna(0.0)

        X = chunk[ALL_FEATS].to_numpy()
        y_true = chunk[TARGET].to_numpy()

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_true)
        if not mask.any():
            continue
        X = X[mask]; y_true = y_true[mask]

        # track true stats in original units
        n = len(y_true)
        eval_n += n
        sum_y += y_true.sum()
        sum_y2 += (y_true**2).sum()

        # standardize features, predict, destandardize predictions
        X_std = (X - feat_mean) / feat_std
        X_t = torch.tensor(X_std, dtype=torch.float32, device=DEVICE)
        pred_std = model(X_t).cpu().numpy().ravel()
        y_pred = pred_std * t_std + t_mean   # back to original C_MID units

        err = y_pred - y_true
        sse += (err**2).sum()
        sae += np.abs(err).sum()

# compute metrics
rmse = np.sqrt(sse / max(eval_n, 1))
mae  = sae / max(eval_n, 1)

# R^2 in original units
den = (sum_y2 - (sum_y**2) / max(eval_n, 1))
if den <= 0:
    r2 = float("nan")
else:
    r2 = 1.0 - (sse / den)

print("\n===== FINAL EVALUATION (on full dataset) =====")
print(f"Samples used          : {eval_n:,}")
print(f"RMSE (dollars)        : {rmse:.4f}")
print(f"MAE  (dollars)        : {mae:.4f}")
print(f"R^2                   : {r2:.4f}")

print("Training + evaluation complete, saving model and stats...")

# save model weights
torch.save(model.state_dict(), "mlp_spx_mid.pth")

# save normalization stats used during training
np.savez(
    "mlp_norm_stats.npz",
    feat_mean=feat_mean,
    feat_std=feat_std,
    t_mean=t_mean,
    t_std=t_std,
)

print("Saved mlp_spx_mid.pth and mlp_norm_stats.npz")

