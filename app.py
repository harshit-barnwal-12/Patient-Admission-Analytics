from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="Patient Admission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #1d4e89 60%, #2f80a8 100%);
        color: white; margin-bottom: 1.1rem;
    }
    .hero h1 {font-size: 2.25rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {font-size: 1rem; margin: 0; color: #e8f2ff;}
    .small-note {font-size: .85rem; color: #6b7280;}
    div[data-testid="stMetric"] {background: #f6f9fc; padding: 14px 16px; border-radius: 14px; border: 1px solid #e5edf5;}
    div[data-testid="stMetric"] label {color: #334e68;}
    .section-head {font-size: 1.25rem; font-weight: 700; margin: .4rem 0 .8rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
def find_csv(filename):
    p = DATA_DIR / filename
    return p if p.exists() else None

def parse_dates(series):
    # Handles common day/month/year and month/day/year strings in the supplied HDHI data.
    # The original dataset is primarily month/day/year, but infer mixed values safely.
    s = series.astype("string").str.strip()
    out = pd.to_datetime(s, errors="coerce", format="mixed")
    if out.isna().mean() > 0.25:
        alt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        out = out.fillna(alt)
    return out

def clean_colnames(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def make_demo_data(n=900, seed=42):
    """Clearly labelled synthetic fallback so the app runs before local CSVs are added."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 730, size=n), unit="D")
    ages = np.clip(rng.normal(58, 17, n), 18, 96).round().astype(int)
    stay = np.clip(rng.gamma(2.2, 2.5, n), 1, 35).round(1)
    discharge = dates + pd.to_timedelta(stay.round().astype(int), unit="D")
    gender = rng.choice(["M", "F"], n, p=[0.61, 0.39])
    outcome = rng.choice(["DISCHARGE", "EXPIRY", "DAMA"], n, p=[0.87, 0.07, 0.06])
    admission_type = rng.choice(["EMERGENCY", "OPD"], n, p=[0.72, 0.28])
    departments = rng.choice(["Cardiology", "Medicine", "Emergency", "Neurology", "Other"], n)
    demo = pd.DataFrame({
        "SNO": np.arange(1, n + 1), "D.O.A": dates.strftime("%m/%d/%Y"),
        "D.O.D": discharge.strftime("%m/%d/%Y"), "AGE": ages, "GENDER": gender,
        "TYPE OF ADMISSION-EMERGENCY/OPD": admission_type, "DURATION OF STAY": stay,
        "OUTCOME": outcome, "DEMO DEPARTMENT": departments,
        "BNP": np.where(rng.random(n) < .45, np.nan, rng.normal(350, 180, n).clip(10)),
        "EF": np.where(rng.random(n) < .10, np.nan, rng.normal(45, 12, n).clip(10, 75)),
        "DM": rng.choice([0, 1], n, p=[.65, .35]), "HTN": rng.choice([0, 1], n, p=[.55, .45]),
        "CAD": rng.choice([0, 1], n, p=[.68, .32]), "CKD": rng.choice([0, 1], n, p=[.83, .17]),
        "HEART FAILURE": rng.choice([0, 1], n, p=[.58, .42]),
    })
    return demo

def load_admissions():
    p = find_csv("HDHI Admission data.csv")
    if p is None:
        return make_demo_data(), True, "Synthetic demo data"
    try:
        df = pd.read_csv(p, low_memory=False)
        return clean_colnames(df), False, "HDHI Admission data.csv"
    except Exception as exc:
        st.error(f"Could not read {p.name}: {exc}")
        return make_demo_data(), True, "Synthetic demo data (CSV read failed)"

def load_optional(filename):
    p = find_csv(filename)
    if p is None:
        return None
    try:
        return clean_colnames(pd.read_csv(p, low_memory=False))
    except Exception:
        return None

def numeric_col(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")

def standardize(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "D.O.A" in df.columns:
        df["Admission Date"] = parse_dates(df["D.O.A"])
    elif "Admission Date" in df.columns:
        df["Admission Date"] = parse_dates(df["Admission Date"])
    else:
        df["Admission Date"] = pd.NaT

    if "D.O.D" in df.columns:
        df["Discharge Date"] = parse_dates(df["D.O.D"])
    elif "Discharge Date" in df.columns:
        df["Discharge Date"] = parse_dates(df["Discharge Date"])
    else:
        df["Discharge Date"] = pd.NaT

    if "AGE" in df.columns:
        df["Age"] = numeric_col(df, "AGE")
    elif "Age" not in df.columns:
        df["Age"] = np.nan

    gender_col = next((c for c in df.columns if c.strip().upper() in ["GENDER", "GENDER "]), None)
    df["Gender"] = df[gender_col].astype("string").str.strip().str.upper() if gender_col else "Unknown"

    outcome_col = next((c for c in df.columns if c.strip().upper() == "OUTCOME"), None)
    df["Outcome"] = df[outcome_col].astype("string").str.strip().str.upper() if outcome_col else "Unknown"

    stay_col = next((c for c in df.columns if c.strip().upper() == "DURATION OF STAY"), None)
    if stay_col:
        df["Length of Stay (days)"] = pd.to_numeric(df[stay_col], errors="coerce")
    else:
        df["Length of Stay (days)"] = (df["Discharge Date"] - df["Admission Date"]).dt.days

    admission_col = next((c for c in df.columns if "TYPE OF ADMISSION" in c.upper()), None)
    df["Admission Type"] = df[admission_col].astype("string").str.strip().str.upper() if admission_col else "Unknown"

    if "D.O.A" not in df.columns and "Admission Date" in df.columns:
        df["D.O.A"] = df["Admission Date"]
    return df

def first_existing(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

# ---------- Load data ----------
raw, is_demo, data_source = load_admissions()
df = standardize(raw)
pollution = load_optional("HDHI Pollution Data.csv")
mortality = load_optional("HDHI Mortality Data.csv")

st.markdown("""
<div class="hero">
  <h1>Patient Admission Analytics System</h1>
  <p>Healthcare data exploration • Admission trends • Patient outcomes • Length of stay</p>
</div>
""", unsafe_allow_html=True)

if is_demo:
    st.info("Demo mode is active. The charts use clearly labelled synthetic example data. To use your real dataset, place `HDHI Admission data.csv` inside the project's `data` folder and restart the app.")
else:
    st.success(f"Loaded local dataset: {data_source}. Patient-level files are read from your computer and are not uploaded by this app.")

# ---------- Sidebar filters ----------
st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Filter the analysis below")
date_series = df["Admission Date"].dropna()
if not date_series.empty:
    min_date, max_date = date_series.min().date(), date_series.max().date()
    date_range = st.sidebar.date_input("Admission date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_range = None

age_values = df["Age"].dropna()
if not age_values.empty:
    age_min = int(max(0, np.floor(age_values.min())))
    age_max = int(np.ceil(age_values.max()))
    age_range = st.sidebar.slider("Patient age", age_min, max(age_min + 1, age_max), (age_min, age_max))
else:
    age_range = None

genders = sorted([str(x) for x in df["Gender"].dropna().unique() if str(x) not in ["<NA>", "nan", ""]])
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
outcomes = sorted([str(x) for x in df["Outcome"].dropna().unique() if str(x) not in ["<NA>", "nan", ""]])
selected_outcomes = st.sidebar.multiselect("Outcome", outcomes, default=outcomes)
types = sorted([str(x) for x in df["Admission Type"].dropna().unique() if str(x) not in ["<NA>", "nan", ""]])
selected_types = st.sidebar.multiselect("Admission type", types, default=types)

filtered = df.copy()
if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if filtered["Admission Date"].notna().any():
        filtered = filtered[filtered["Admission Date"].between(start_date, end_date) | filtered["Admission Date"].isna()]
if age_range is not None:
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1]) | filtered["Age"].isna()]
if selected_genders:
    filtered = filtered[filtered["Gender"].astype(str).isin(selected_genders)]
if selected_outcomes:
    filtered = filtered[filtered["Outcome"].astype(str).isin(selected_outcomes)]
if selected_types:
    filtered = filtered[filtered["Admission Type"].astype(str).isin(selected_types)]

# ---------- KPI row ----------
total = len(filtered)
avg_age = filtered["Age"].mean()
avg_stay = filtered["Length of Stay (days)"].replace([np.inf, -np.inf], np.nan).clip(lower=0).mean()
outcome_clean = filtered["Outcome"].astype(str).str.upper()
expiry_count = int(outcome_clean.str.contains("EXPIR|DEATH|DIED", regex=True, na=False).sum())
discharge_count = int(outcome_clean.str.contains("DISCHARGE", regex=True, na=False).sum())
expiry_rate = (expiry_count / total * 100) if total else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Admissions in view", f"{total:,}")
k2.metric("Average age", f"{avg_age:.1f} years" if pd.notna(avg_age) else "N/A")
k3.metric("Average length of stay", f"{avg_stay:.1f} days" if pd.notna(avg_stay) else "N/A")
k4.metric("Recorded expiry outcomes", f"{expiry_count:,}", f"{expiry_rate:.1f}% of filtered records")

st.caption("KPIs update when sidebar filters change. Recorded outcome labels reflect the dataset and should not be interpreted as a clinical quality measure.")

# ---------- Tabs ----------
tab_overview, tab_outcomes, tab_clinical, tab_data = st.tabs(["Overview", "Outcomes & Demographics", "Clinical Indicators", "Data Quality & Export"])

with tab_overview:
    left, right = st.columns((1.5, 1))
    with left:
        st.subheader("Admissions over time")
        dated = filtered.dropna(subset=["Admission Date"]).copy()
        if not dated.empty:
            dated["Month"] = dated["Admission Date"].dt.to_period("M").dt.to_timestamp()
            monthly = dated.groupby("Month").size().reset_index(name="Admissions")
            fig = px.line(monthly, x="Month", y="Admissions", markers=True,
                          title="Monthly admission volume",
                          color_discrete_sequence=["#1976D2"])
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_title="", yaxis_title="Admissions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid admission dates are available for this selection.")
    with right:
        st.subheader("Admission outcomes")
        outcome_counts = filtered["Outcome"].fillna("Unknown").astype(str).replace("<NA>", "Unknown").value_counts().rename_axis("Outcome").reset_index(name="Records")
        fig = px.pie(outcome_counts, names="Outcome", values="Records", hole=.55,
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=5, r=5, t=15, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Age distribution")
        ages = filtered.dropna(subset=["Age"])
        if not ages.empty:
            fig = px.histogram(ages, x="Age", nbins=24, color_discrete_sequence=["#2A9D8F"])
            fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Age (years)", yaxis_title="Patient records")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Age data is unavailable.")
    with right:
        st.subheader("Length of stay")
        los = filtered["Length of Stay (days)"].dropna()
        los = los[(los >= 0) & (los <= los.quantile(.99) if len(los) else True)]
        if not los.empty:
            fig = px.histogram(los, x="Length of Stay (days)", nbins=25, color_discrete_sequence=["#F4A261"])
            fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Days", yaxis_title="Patient records")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Length-of-stay data is unavailable.")

with tab_outcomes:
    left, right = st.columns(2)
    with left:
        st.subheader("Outcome by gender")
        tmp = filtered.copy()
        tmp["Outcome"] = tmp["Outcome"].fillna("Unknown").astype(str)
        tmp["Gender"] = tmp["Gender"].fillna("Unknown").astype(str)
        cross = tmp.groupby(["Gender", "Outcome"]).size().reset_index(name="Records")
        if not cross.empty:
            fig = px.bar(cross, x="Gender", y="Records", color="Outcome", barmode="stack",
                         color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), xaxis_title="", yaxis_title="Records")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outcome data available.")
    with right:
        st.subheader("Admission type")
        typ = filtered["Admission Type"].fillna("Unknown").astype(str).value_counts().rename_axis("Admission Type").reset_index(name="Records")
        fig = px.bar(typ, x="Admission Type", y="Records", text="Records", color="Admission Type",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), showlegend=False, xaxis_title="", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average length of stay by outcome")
    stay_outcome = filtered.dropna(subset=["Length of Stay (days)"]).groupby("Outcome", dropna=False)["Length of Stay (days)"].mean().reset_index()
    if not stay_outcome.empty:
        fig = px.bar(stay_outcome, x="Outcome", y="Length of Stay (days)", text_auto=".1f",
                     color="Outcome", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), showlegend=False, xaxis_title="", yaxis_title="Average days")
        st.plotly_chart(fig, use_container_width=True)

with tab_clinical:
    st.subheader("Selected clinical indicators")
    candidates = [c for c in ["HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "BNP", "EF"] if c in filtered.columns]
    if candidates:
        chosen = st.multiselect("Choose numeric indicators", candidates, default=candidates[:4])
        if chosen:
            clinical = filtered[chosen].apply(pd.to_numeric, errors="coerce")
            desc = clinical.describe().T.reset_index().rename(columns={"index": "Indicator"})
            st.dataframe(desc.round(2), use_container_width=True, hide_index=True)
            if len(chosen) >= 2:
                corr = clinical.corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1, title="Correlation between selected indicators")
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose one or more indicators above.")
    else:
        st.info("No recognized clinical measurement columns were found.")

    st.subheader("Common recorded conditions")
    conditions = [c for c in ["DM", "HTN", "CAD", "CKD", "HEART FAILURE", "ACS", "AKI", "ANAEMIA", "CVA INFRACT"] if c in filtered.columns]
    if conditions:
        condition_counts = []
        for c in conditions:
            vals = filtered[c].astype(str).str.strip().str.upper()
            positives = vals.isin(["1", "1.0", "YES", "Y", "TRUE", "PRESENT"])
            condition_counts.append({"Condition": c, "Records marked positive": int(positives.sum())})
        cond_df = pd.DataFrame(condition_counts).sort_values("Records marked positive", ascending=True)
        fig = px.bar(cond_df, x="Records marked positive", y="Condition", orientation="h",
                     color_discrete_sequence=["#4776A8"])
        fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Records", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Condition columns were not found in the current dataset.")

with tab_data:
    st.subheader("Data quality summary")
    total_cells = raw.shape[0] * raw.shape[1]
    missing_cells = int(raw.isna().sum().sum())
    dup_rows = int(raw.duplicated().sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Raw rows", f"{len(raw):,}")
    q2.metric("Raw columns", f"{raw.shape[1]:,}")
    q3.metric("Duplicate rows", f"{dup_rows:,}")
    st.write(f"Missing cells: **{missing_cells:,} / {total_cells:,}** ({missing_cells / total_cells * 100:.1f}%)" if total_cells else "No data available.")
    missing = raw.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(20).rename_axis("Column").reset_index(name="Missing values")
    if not missing.empty:
        fig = px.bar(missing.sort_values("Missing values"), x="Missing values", y="Column", orientation="h",
                     title="Columns with the most missing values", color_discrete_sequence=["#D97757"])
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values were found in the raw data.")

    st.subheader("Filtered records")
    # Hide direct patient identifiers from the dashboard table.
    hide_cols = {c for c in filtered.columns if c.strip().upper() in {"MRD NO.", "MRD", "SNO", "S.NO"}}
    visible = filtered[[c for c in filtered.columns if c not in hide_cols]].copy()
    st.dataframe(visible.head(500), use_container_width=True, hide_index=True)
    st.caption("Table preview is limited to 500 rows. Direct record identifiers are hidden.")

    csv_bytes = visible.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered table as CSV", data=csv_bytes,
                       file_name="patient_admission_filtered_export.csv", mime="text/csv")

# Optional pollution analysis, only if file is present.
with st.expander("Optional: Pollution data overview", expanded=False):
    if pollution is None:
        st.info("To enable this section, place `HDHI Pollution Data.csv` in the local data folder.")
    else:
        p = pollution.copy()
        date_col = first_existing(p, ["DATE", "Date"])
        aqi_col = first_existing(p, ["AQI", "Aqi"])
        if date_col and aqi_col:
            p["Pollution Date"] = parse_dates(p[date_col])
            p["AQI numeric"] = pd.to_numeric(p[aqi_col], errors="coerce")
            p = p.dropna(subset=["Pollution Date", "AQI numeric"])
            if not p.empty:
                p["Month"] = p["Pollution Date"].dt.to_period("M").dt.to_timestamp()
                pm = p.groupby("Month")["AQI numeric"].mean().reset_index()
                fig = px.line(pm, x="Month", y="AQI numeric", markers=True, title="Average monthly AQI",
                              color_discrete_sequence=["#7C3AED"])
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_title="", yaxis_title="Average AQI")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("This is a descriptive time trend. It does not establish that pollution caused any patient outcome.")
        else:
            st.warning("Pollution file was found, but expected DATE/AQI columns were not recognized.")

st.markdown("---")
st.caption("Academic project prototype • SDG 3: Good Health and Well-being • For educational analysis only, not for clinical decision-making.")
