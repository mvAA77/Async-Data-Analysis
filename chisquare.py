import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# =========================
# CONFIG: update these
# =========================
CSV_PATH = "async_data.csv"   # <-- change to your csv filename/path
STUDENT_ID_COL = "student_id" # <-- change if your column name differs (e.g., "user_id")
SAT_COL = "student_satisfaction_rating"  # <-- change if your column name differs (e.g., "satisfaction", "E_rating")
DRAFT_COL = "draft"     # optional; can help order essays if present

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

# Basic cleanup
df = df.copy()
df[SAT_COL] = df[SAT_COL].astype(str).str.strip()

# Keep only rows with satisfaction ratings of interest
valid = df[SAT_COL].isin(["E+", "E", "E-"])
df = df.loc[valid].copy()

# Sort by draft number instead of submission time
df[DRAFT_COL] = pd.to_numeric(df[DRAFT_COL], errors="coerce")

df = df.sort_values([STUDENT_ID_COL, DRAFT_COL])


# =========================
# FEATURE: essay number per student
# =========================
df["essay_num"] = df.groupby(STUDENT_ID_COL).cumcount() + 1

# Binary outcome: E+ vs not E+
df["is_E_plus"] = (df[SAT_COL] == "E+").astype(int)

# =========================
# 1) Quick descriptive view
# =========================
summary = (
    df.groupby("essay_num")
      .agg(n=("is_E_plus", "size"), eplus_rate=("is_E_plus", "mean"))
      .reset_index()
)

# Bucket for stability (helps if essay_num gets large / sparse)
def bucket(n):
    if 1 <= n <= 2: return "1-2"
    if 3 <= n <= 6: return "3-6"
    if 7 <= n <= 11: return "6-11"
    return "11+"

df["essay_bucket"] = df["essay_num"].apply(bucket)

bucket_summary = (
    df.groupby("essay_bucket")
      .agg(n=("is_E_plus", "size"), eplus_rate=("is_E_plus", "mean"))
      .reindex(["1-2","3-6","6-11", "11+"])
      .reset_index()
)

print("\nE+ rate by essay_num (raw):")
print(summary.head(15).to_string(index=False))

print("\nE+ rate by bucket:")
print(bucket_summary.to_string(index=False))

# =========================
# 2) Hypothesis test: first essay vs later essays
# =========================
first = df.loc[df["essay_num"] == 1, "is_E_plus"]
later = df.loc[df["essay_num"] >= 2, "is_E_plus"]

# Two-proportion z-test (E+ rate difference)
count = np.array([first.sum(), later.sum()])
nobs  = np.array([first.size, later.size])

z_stat, p_val = sm.stats.proportions_ztest(count, nobs)
diff = first.mean() - later.mean()

print("\nTwo-proportion z-test: essay #1 vs essays #2+")
print(f"  E+ rate (essay #1):  {first.mean():.4f}  (n={first.size})")
print(f"  E+ rate (essay #2+): {later.mean():.4f}  (n={later.size})")
print(f"  Difference (1 - 2+): {diff:.4f}")
print(f"  z = {z_stat:.3f}, p = {p_val:.4g}")

# =========================
# 3) Trend test: does E+ probability drop as essay_num increases?
#    Logistic regression: is_E_plus ~ essay_num
# =========================
X = sm.add_constant(df["essay_num"])
model = sm.Logit(df["is_E_plus"], X).fit(disp=0)

coef = model.params["essay_num"]
se = model.bse["essay_num"]
odds_ratio = np.exp(coef)

print("\nLogistic regression: is_E_plus ~ essay_num")
print(f"  coef(essay_num) = {coef:.4f} (SE={se:.4f})")
print(f"  odds ratio per +1 essay = {odds_ratio:.4f}")
print(model.summary().tables[1])

# =========================
# 4) Logistic regression with clustered SE by student
# =========================
X = sm.add_constant(df["essay_num"])

cluster_model = sm.Logit(df["is_E_plus"], X).fit(
    disp=0,
    cov_type="cluster",
    cov_kwds={"groups": df[STUDENT_ID_COL]}
)

c_coef = cluster_model.params["essay_num"]
c_se = cluster_model.bse["essay_num"]
c_or = np.exp(c_coef)

print("\nLogistic regression with clustered SE by student")
print(f"  coef(essay_num) = {c_coef:.4f} (clustered SE={c_se:.4f})")
print(f"  odds ratio per +1 essay = {c_or:.4f}")
print(cluster_model.summary().tables[1])

