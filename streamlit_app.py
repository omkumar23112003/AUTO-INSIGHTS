
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from io import BytesIO
from typing import Tuple, Dict, List

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, r2_score, mean_absolute_error

st.set_page_config(page_title="Auto Insights ML", layout="wide")

# ------------------------------
# Utility functions
# ------------------------------

def load_data(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded, engine="openpyxl")

def detect_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    # Try to infer datetimes
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                # accept as datetime if at least 80% parse success (raise would fail on any bad)
                df[col] = parsed
            except Exception:
                pass
    dtypes = {
        "datetime": [],
        "numeric": [],
        "categorical": []
    }
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            dtypes["datetime"].append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            dtypes["numeric"].append(col)
        else:
            dtypes["categorical"].append(col)
    return dtypes

def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = []
    for col in df.columns:
        col_data = df[col]
        miss = col_data.isna().mean()
        entry = {
            "column": col,
            "dtype": str(col_data.dtype),
            "missing_pct": round(100*miss, 2),
            "n_unique": col_data.nunique(dropna=True)
        }
        if np.issubdtype(col_data.dtype, np.number):
            entry.update({
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "p25": col_data.quantile(0.25),
                "p50": col_data.median(),
                "p75": col_data.quantile(0.75),
                "max": col_data.max()
            })
        profile.append(entry)
    return pd.DataFrame(profile)

def safe_fig():
    fig = plt.figure()
    return fig

def plot_basic_charts(df: pd.DataFrame, types: Dict[str, List[str]]):
    # Missingness bar
    st.subheader("Data Quality")
    miss = df.isna().mean().sort_values(ascending=False)
    fig = safe_fig()
    miss.plot(kind="bar")
    plt.title("Missingness by Column")
    plt.xlabel("Column")
    plt.ylabel("Fraction Missing")
    st.pyplot(fig, clear_figure=True)

    # Correlation heatmap (numeric only)
    if len(types["numeric"]) >= 2:
        st.subheader("Correlation (numeric columns)")
        corr = df[types["numeric"]].corr(numeric_only=True)
        fig = safe_fig()
        im = plt.imshow(corr.values, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)

    # Time series line charts for datetime + numeric
    if types["datetime"] and types["numeric"]:
        dt = types["datetime"][0]
        df_sorted = df.sort_values(dt)
        st.subheader(f"Time Series (by {dt})")
        for num in types["numeric"][:5]:
            fig = safe_fig()
            plt.plot(df_sorted[dt], df_sorted[num])
            plt.title(f"{num} over time")
            plt.xlabel(dt)
            plt.ylabel(num)
            st.pyplot(fig, clear_figure=True)

    # Top categories for first categorical vs first numeric
    if types["categorical"] and types["numeric"]:
        cat = types["categorical"][0]
        num = types["numeric"][0]
        st.subheader(f"Category Comparison: {cat} vs {num}")
        top = df.groupby(cat)[num].mean().sort_values(ascending=False).head(15)
        fig = safe_fig()
        top.plot(kind="bar")
        plt.title(f"Average {num} by {cat}")
        plt.xlabel(cat)
        plt.ylabel(f"Mean {num}")
        st.pyplot(fig, clear_figure=True)

def generate_rule_based_insights(df: pd.DataFrame, types: Dict[str, List[str]]) -> List[str]:
    insights = []
    # Missingness insight
    miss = df.isna().mean().sort_values(ascending=False)
    top_miss = miss.head(3)[miss.head(3) > 0]
    if not top_miss.empty:
        parts = [f"{col}: {pct:.1%}" for col, pct in top_miss.items()]
        insights.append("Highest missingness → " + ", ".join(parts))

    # Correlation insight
    if len(types["numeric"]) >= 2:
        corr = df[types["numeric"]].corr(numeric_only=True)
        tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        tri_vals = tri.unstack().dropna()
        if not tri_vals.empty:
            pair, val = tri_vals.abs().sort_values(ascending=False).index[0], tri_vals.abs().max()
            col_a, col_b = pair
            insights.append(f"Strongest numeric association: {col_a} ↔ {col_b} (|corr| ≈ {val:.2f}).")

    # Category leader insight
    if types["categorical"] and types["numeric"]:
        cat = types["categorical"][0]
        num = types["numeric"][0]
        grp = df.groupby(cat)[num].mean().sort_values(ascending=False)
        if not grp.empty:
            best = grp.index[0]
            worst = grp.index[-1]
            insights.append(f"In {cat}, '{best}' leads and '{worst}' lags on average {num}.")

    # Trend insight (first datetime + first numeric)
    if types["datetime"] and types["numeric"]:
        dt = types["datetime"][0]
        num = types["numeric"][0]
        df_ts = df[[dt, num]].dropna().sort_values(dt)
        if len(df_ts) >= 3:
            x = np.arange(len(df_ts))
            y = df_ts[num].values
            # Simple slope using linear fit
            slope = np.polyfit(x, y, 1)[0]
            direction = "increasing" if slope > 0 else "decreasing"
            insights.append(f"Time trend: {num} appears {direction} overall (linear slope {slope:.3g}).")

    return insights

def run_clustering(df: pd.DataFrame, types: Dict[str, List[str]]) -> Tuple[pd.Series, List[str]]:
    insights = []
    if len(types["numeric"]) < 2:
        return None, ["Not enough numeric columns for clustering."]
    X = df[types["numeric"]].dropna()
    if len(X) < 10:
        return None, ["Not enough rows for clustering."]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # Pick K via quick elbow proxy (inertia drop)
    max_k = min(8, max(3, int(math.sqrt(len(X)))))
    inertias = []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(Xs)
        inertias.append(km.inertia_)
    # choose k where marginal drop falls below threshold
    deltas = np.diff(inertias)
    if len(deltas) == 0:
        k_opt = 2
    else:
        k_opt = 2 + int(np.argmin(np.abs(deltas)))

    km = KMeans(n_clusters=k_opt, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    # Simple cluster size insight
    sizes = pd.Series(labels).value_counts().sort_index()
    insights.append(f"Clustering found {k_opt} groups with sizes: " + ", ".join([f"C{i}={n}" for i, n in sizes.items()]))
    return pd.Series(labels, index=X.index, name="cluster"), insights

def run_anomaly(df: pd.DataFrame, types: Dict[str, List[str]]) -> Tuple[pd.Series, List[str]]:
    if len(types["numeric"]) < 2:
        return None, ["Not enough numeric columns for anomaly detection."]
    X = df[types["numeric"]].dropna()
    if len(X) < 20:
        return None, ["Not enough rows for anomaly detection."]
    iso = IsolationForest(contamination="auto", random_state=42)
    preds = iso.fit_predict(X.values)  # -1 anomaly, 1 normal
    score = iso.score_samples(X.values)
    anomalies = np.where(preds == -1)[0]
    insight = [f"Anomaly detection flagged {len(anomalies)} unusual rows out of {len(X)}."]
    s = pd.Series((preds == -1).astype(int), index=X.index, name="is_anomaly")
    return s, insight

def run_supervised(df: pd.DataFrame, target: str, types: Dict[str, List[str]]) -> Tuple[Dict, List[str]]:
    y = df[target]
    X = df.drop(columns=[target])

    # Encode categoricals
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            X[col] = X[col].astype("category").cat.codes
        elif np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = X[col].astype("int64") // 10**9  # seconds

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan)

    is_classification = (y.dtype == "object") or (y.nunique(dropna=True) <= 20)
    if is_classification:
        y = y.astype("category").cat.codes
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {}
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results["task"] = "classification"
        results["accuracy"] = float(acc)
        results["f1_weighted"] = float(f1)
    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results["task"] = "regression"
        results["r2"] = float(r2)
        results["mae"] = float(mae)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    results["top_features"] = importances.to_dict()

    return results, []

# ------------------------------
# UI
# ------------------------------

st.title("📈 Auto Insights + ML from Excel/CSV")
st.write("Upload a dataset and get instant charts, ML, and natural-language insights.")

uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])

if uploaded:
    with st.spinner("Loading data..."):
        df = load_data(uploaded)

    st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")

    types = detect_types(df)
    profile_df = basic_profile(df)

    st.subheader("Schema & Profile")
    st.dataframe(profile_df, use_container_width=True)

    st.markdown("**Detected types**")
    st.json(types)

    # Charts
    plot_basic_charts(df, types)

    # Insights
    st.subheader("🧠 Meaningful Insights")
    base_ins = generate_rule_based_insights(df, types)
    for ins in base_ins:
        st.write("• " + ins)

    # Clustering
    st.subheader("🤝 Unsupervised: Clustering")
    labels, ins = run_clustering(df, types)
    for i in ins:
        st.write("• " + i)
    if labels is not None:
        tmp = df.copy()
        tmp["cluster"] = labels
        st.dataframe(tmp.head(20), use_container_width=True)

        # Simple 2D projection using first two numeric cols
        nums = types["numeric"]
        if len(nums) >= 2:
            st.write("Cluster scatter (first two numeric features):")
            fig = plt.figure()
            for c in sorted(labels.unique()):
                idx = labels[labels == c].index
                plt.scatter(df.loc[idx, nums[0]], df.loc[idx, nums[1]], label=f"C{c}", alpha=0.7)
            plt.xlabel(nums[0]); plt.ylabel(nums[1]); plt.legend()
            st.pyplot(fig, clear_figure=True)

    # Anomaly detection
    st.subheader("🧭 Unsupervised: Anomaly Detection")
    anom, ins = run_anomaly(df, types)
    for i in ins:
        st.write("• " + i)
    if anom is not None:
        tmp = df.copy()
        tmp["is_anomaly"] = anom
        st.dataframe(tmp[tmp["is_anomaly"] == 1].head(20), use_container_width=True)

    # Supervised learning (optional)
    st.subheader("🎯 Supervised Learning (optional)")
    target = st.selectbox("Select a target column (or leave blank to skip)", [""] + list(df.columns))
    if target:
        with st.spinner("Training model..."):
            results, _ = run_supervised(df, target, types)
        st.write(results)

        # Feature importance bar
        feats = results.get("top_features", {})
        if feats:
            fi = pd.Series(feats)
            fig = plt.figure()
            fi.sort_values(ascending=True).plot(kind="barh")
            plt.title("Top Feature Importances")
            st.pyplot(fig, clear_figure=True)

    st.info("Tip: Use the sidebar ▶ to resize the main area for larger charts.")

else:
    st.info("Upload an Excel or CSV file to begin.")
