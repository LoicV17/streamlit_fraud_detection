import streamlit as st
import pandas as pd
import boto3
import os
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv

# =================================
# CONFIG STREAMLIT
# =================================
st.set_page_config(page_title="Fraud Detection Report", page_icon="🕵️", layout="wide")

# Style
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f5f7fa;
        text-align: center;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        color: #666;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #111;
    }
    </style>
""", unsafe_allow_html=True)

# =================================
# CONFIG AWS
# =================================
load_dotenv()
s3 = boto3.client("s3")
BUCKET = os.getenv("AIRFLOW_S3_BUCKET", "fraud-detection-loicvalentini")
KEY = "reports/full/scored_payments.parquet"

@st.cache_data
def load_data():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return df

df = load_data()

if st.button("🔄 Recharger les données"):
    st.cache_data.clear()
    df = load_data()

# =================================
# CORRECTION DES DATES
# =================================

# 1. Si event_time existe en timestamp numérique → conversion automatique
if "event_time" in df.columns and df["event_time"].dtype in ["int64", "float64"]:
    max_len = df["event_time"].astype(str).str.len().max()

    if max_len == 10:
        df["event_time"] = pd.to_datetime(df["event_time"], unit="s", errors="coerce")
    elif max_len == 13:
        df["event_time"] = pd.to_datetime(df["event_time"] / 1000, unit="s", errors="coerce")
    else:
        df["event_time"] = pd.NaT

# 2. Sinon : reconstruire la date à partir des colonnes year/month/day/hour
elif {"trans_year", "trans_month", "trans_day", "trans_hour"}.issubset(df.columns):
    df["event_time"] = pd.to_datetime(
        df["trans_year"].astype(int).astype(str) + "-" +
        df["trans_month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["trans_day"].astype(int).astype(str).str.zfill(2) + " " +
        df["trans_hour"].astype(int).astype(str).str.zfill(2) + ":" +
        df.get("trans_minute", 0).astype(int).astype(str).str.zfill(2),
        errors="coerce"
    )

# Nettoyage final des dates invalides
df = df[df["event_time"].notna()]

# =================================
# HEADER
# =================================
st.title("🕵️ Rapport Fraude Global")
st.markdown("Un aperçu complet des transactions scorées avec détection de fraude.")

# =================================
# KPIs
# =================================
fraud_amount = df.loc[df["prediction"] == 1, "amt"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Transactions cumulées</div><div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Fraudes cumulées</div><div class='metric-value'>{df['prediction'].sum():,}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Taux global</div><div class='metric-value'>{100*df['prediction'].mean():.2f}%</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Montant total analysé (€)</div><div class='metric-value'>{df['amt'].sum():,.0f}</div></div>", unsafe_allow_html=True)
with col5:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Montant total fraudé (€)</div><div class='metric-value' style='color:red;'>{fraud_amount:,.0f}</div></div>", unsafe_allow_html=True)

st.divider()

# =================================
# VISUELS TEMPORELS
# =================================
st.subheader("📅 Évolution du taux de fraude")

granularity = st.radio("Granularité :", ["Heure", "Jour", "Semaine", "Mois"], horizontal=True)

# Définition de la période
if granularity == "Heure":
    df["period"] = df["event_time"].dt.floor("H")
elif granularity == "Jour":
    df["period"] = df["event_time"].dt.floor("D")
elif granularity == "Semaine":
    df["period"] = df["event_time"].dt.to_period("W").apply(lambda r: r.start_time)
elif granularity == "Mois":
    df["period"] = df["event_time"].dt.to_period("M").apply(lambda r: r.start_time)

df["period"] = pd.to_datetime(df["period"], errors="coerce")
df = df[df["period"].notna()]

fraude_by_period = df.groupby("period")["prediction"].mean().reset_index()
fraude_by_period["fraud_rate"] = fraude_by_period["prediction"] * 100

chart = alt.Chart(fraude_by_period).mark_line(point=True).encode(
    x=alt.X("period:T", title="Période"),
    y=alt.Y("fraud_rate:Q", title="Taux fraude (%)"),
    tooltip=["period", alt.Tooltip("fraud_rate:Q", format=".2f")]
).properties(height=400)

st.altair_chart(chart, use_container_width=True)

# =================================
# VISUELS ANALYTIQUES
# =================================
st.subheader("🔎 Analyse des fraudes")

col1, col2 = st.columns(2)

with col1:
    fraude_cat = df[df["prediction"] == 1]["category"].value_counts().reset_index()
    fraude_cat.columns = ["category", "count"]

    fig1, ax1 = plt.subplots()
    ax1.pie(fraude_cat["count"], labels=fraude_cat["category"], autopct="%1.1f%%")
    ax1.set_title("Répartition des fraudes par catégorie")
    st.pyplot(fig1)

with col2:
    US_STATES = { ... }  # (garde ton mapping ici)

    df["state_full"] = df["state"].map(US_STATES).fillna(df["state"])

    fraudes_par_state = df[df["prediction"] == 1]["state_full"].value_counts().reset_index()
    fraudes_par_state.columns = ["État", "Nombre de fraudes"]

    chart_state = alt.Chart(fraudes_par_state).mark_bar(color="red").encode(
        x="Nombre de fraudes:Q",
        y=alt.Y("État:N", sort="-x"),
        tooltip=["État", "Nombre de fraudes"]
    ).properties(height=350)

    st.altair_chart(chart_state, use_container_width=True)

# =================================
# DATASET COMPLET
# =================================
df_display = df.copy()
df_display["date"] = df_display["event_time"].dt.strftime("%Y-%m-%d %H:%M")

st.subheader("📂 Détails des fraudes détectées")
fraude_details = df_display[df_display["prediction"] == 1]
st.dataframe(fraude_details)

csv = fraude_details.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger CSV complet", data=csv, file_name="fraudes.csv", mime="text/csv")

# =================================
# SIDEBAR
# =================================
with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<span style='color:gray; font-size:12px;'>"
        "Made by <b>Loic Valentini</b><br>"
        "Jedha AIA – Projet Bloc 3 – <i>Fraud_detection</i>"
        "</span>",
        unsafe_allow_html=True
    )
