import os, io, ast, sqlite3, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from src.t_y_gen import assign_treatments_tcga, generate_tcga_outcomes


np.random.seed(42)


con = sqlite3.connect("./tcga.db")   
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';",
    con
)
print(tables)

df_clinical   = pd.read_sql("SELECT * FROM clinical;",    con)
df_rnaseq_raw = pd.read_sql("SELECT * FROM rnaseq;",      con)

assert {"id", "data"}.issubset(df_rnaseq_raw.columns)
df_rnaseq = df_rnaseq_raw.loc[:, ["id", "data"]].copy()


for df in (df_clinical, df_rnaseq):
    df["id"] = df["id"].astype(str).str.strip().str.upper()
df_clinical["id12"] = df_clinical["id"].str[:12]
df_rnaseq["id12"]   = df_rnaseq["id"].str[:12]

print("n_clinical =", len(df_clinical), "n_rnaseq =", len(df_rnaseq))
print("overlap(12-char) =", len(set(df_clinical["id12"]).intersection(set(df_rnaseq["id12"]))))


data = pd.merge(df_clinical, df_rnaseq, on="id12", how="inner", suffixes=("_clin", "_rna"))
print("after merge rows =", len(data))
if len(data) == 0:
    raise RuntimeError("临床与 rnaseq 的 id 没有重叠。请检查数据库并确认两表使用相同的 barcode 规则（前12位/全长）。")

encoded = []
for i in range(len(data)):
    byte_data = ast.literal_eval(str(data.loc[i, "data"]))
    arr = np.load(io.BytesIO(byte_data))   # shape: (G,)
    encoded.append(arr)


encoded = pd.DataFrame(encoded)
encoded.columns = [f"X_{j}" for j in range(encoded.shape[1])]


data = data.drop(columns=["data"])
data = pd.concat([data.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)

print("data shape (with clin+genes) =", data.shape)


gene_cols = [c for c in data.columns if c.startswith("X_")]

clin_cols = [c for c in data.columns if (c not in gene_cols) and (c not in ["id_clin","id_rna","id12","id"])]

data_clin = data[clin_cols].copy()
for col in data_clin.select_dtypes(include=["object", "category"]).columns:
    data_clin[col], _ = pd.factorize(data_clin[col])
constant_cols = [c for c in data_clin.columns if data_clin[c].nunique() == 1]
if constant_cols:
    data_clin.drop(columns=constant_cols, inplace=True)


variances = data[gene_cols].var(axis=0)
top_4000 = variances.sort_values(ascending=False).head(4000).index.tolist()
print("genes picked =", len(top_4000))


X_full = pd.concat([data_clin.reset_index(drop=True),
                    data[top_4000].reset_index(drop=True)], axis=1).astype(np.float32)

print("X_full shape =", X_full.shape)

idx_all = np.arange(len(X_full))
if len(idx_all) == 0:
    raise RuntimeError("X_full 行数为 0，无法切分。请回看上面的行数打印。")

idx_trv, idx_te = train_test_split(idx_all, test_size=0.20, random_state=42, shuffle=True)
val_rel = 0.12 / 0.80
idx_tr, idx_va = train_test_split(idx_trv, test_size=val_rel, random_state=42, shuffle=True)


scaler = StandardScaler().fit(X_full.iloc[idx_tr])
X_tr = pd.DataFrame(scaler.transform(X_full.iloc[idx_tr]), columns=X_full.columns)
X_va = pd.DataFrame(scaler.transform(X_full.iloc[idx_va]), columns=X_full.columns)
X_te = pd.DataFrame(scaler.transform(X_full.iloc[idx_te]), columns=X_full.columns)


normalizer = Normalizer(norm="l2")
X_tr = pd.DataFrame(normalizer.fit_transform(X_tr), columns=X_full.columns)
X_va = pd.DataFrame(normalizer.transform(X_va),   columns=X_full.columns)
X_te = pd.DataFrame(normalizer.transform(X_te),   columns=X_full.columns)


X_scaled_full = np.zeros_like(X_full.values, dtype=np.float32)
X_scaled_full[idx_tr] = X_tr.values
X_scaled_full[idx_va] = X_va.values
X_scaled_full[idx_te] = X_te.values

X_scaled_full_df = pd.DataFrame(X_scaled_full, columns=X_full.columns, dtype=np.float32)

treatments, v2, v3 = assign_treatments_tcga(X_scaled_full_df)
y, v1, v2, v3 = generate_tcga_outcomes(X_scaled_full_df, treatments.reshape(-1, 1))


t_tr = treatments[idx_tr].astype(np.float32)
t_va = treatments[idx_va].astype(np.float32)
t_te = treatments[idx_te].astype(np.float32)
y_tr = y[idx_tr].astype(np.float32)
y_va = y[idx_va].astype(np.float32)
y_te = y[idx_te].astype(np.float32)


N_GRID = 65
t_grid = np.linspace(0.0, 1.0, N_GRID, dtype=np.float32).reshape(-1, 1)

X_te_np = X_te.values.astype(np.float32)
T_eval  = np.tile(t_grid, (X_te_np.shape[0], 1))                    # (n_test*N_GRID, 1)
X_eval  = np.repeat(X_te_np, N_GRID, axis=0)                        # (n_test*N_GRID, d)
core    = (X_eval @ v1) + 12.0*(X_eval @ v2)*T_eval.ravel() - 12.0*(X_eval @ v3)*(T_eval.ravel()**2)
Y_eval  = (10.0 * core).astype(np.float32)
eval_test = np.hstack([T_eval, Y_eval.reshape(-1, 1)]).reshape(-1, N_GRID, 2)  # (n_test,65,2)

OUT_DIR = os.path.join("data", "TCGA")
os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "train.npy"),
        np.hstack([X_tr.values, t_tr.reshape(-1,1), y_tr.reshape(-1,1)]).astype(np.float32))
# 如需单独验证集可另存
# np.save(os.path.join(OUT_DIR, "val.npy"),
#         np.hstack([X_va.values, t_va.reshape(-1,1), y_va.reshape(-1,1)]).astype(np.float32))
np.save(os.path.join(OUT_DIR, "test.npy"),
        np.hstack([X_te.values, t_te.reshape(-1,1), y_te.reshape(-1,1)]).astype(np.float32))
np.save(os.path.join(OUT_DIR, "eval_test.npy"), eval_test.astype(np.float32))
np.save(os.path.join(OUT_DIR, "v_vector.npy"),  np.stack([v1, v2, v3], axis=0))

with open(os.path.join(OUT_DIR, "info.pkl"), "wb") as f:
    pickle.dump({
        "split": {"train": 0.68, "val": 0.12, "test": 0.20},
        "n_grid_1dim": N_GRID,
        "dim_treat": 1,
        "selection_bias": 2,
        "beta_const": 2,
    }, f)

print("✔ Saved to", OUT_DIR)

