import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import time
from locale import setlocale, LC_NUMERIC, atof
import locale
import re

# Configuration initiale
try:
    setlocale(LC_NUMERIC, 'en_US.UTF-8')
except:
    setlocale(LC_NUMERIC, '')

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="Manager Risk Analysis",
    page_icon="📈",
)

# Style CSS
st.markdown("""
    <style>
        .stApp {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #333333;
        }
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.3em;
        }
        .metric-container {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 1.5em;
            margin-bottom: 1.5em;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5em 1em;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #1a2634;
        }
        .manager-list {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 4px;
        }
        .metric-type-btn {
            font-size: 0.8em;
            padding: 0.2em 0.5em;
            margin: 0.1em;
        }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown("<h1 style='text-align: center; margin-bottom: 1em;'>Manager Risk-Reward Analysis</h1>", unsafe_allow_html=True)

# Fonctions utilitaires
def clean_european_number(value):
    if pd.isna(value) or value == '': return 0.0
    if isinstance(value, str):
        value = value.strip().replace(' ', '').replace('.', '').replace(',', '.')
    try: return float(value)
    except: return 0.0

def clean_numeric(value, is_manager_name=False):
    if is_manager_name: return str(value).strip()
    if pd.isna(value) or value == '': return 0.0
    if isinstance(value, str):
        value = value.strip().strip('"').replace(' ', '')
        if '%' in value:
            try: return clean_european_number(value.replace('%', '')) / 100
            except: return 0.0
        return clean_european_number(value)
    return float(value)

def min_max_scale(series, invert=False):
    numeric = series.copy()
    mn, mx = numeric.min(), numeric.max()
    if invert: numeric = mx - numeric + mn
    scaled = (numeric - mn) / (mx - mn) if (mx != mn) else 0.5
    return scaled.round(6)

def calculate_bubble_size(deal_count, aum):
    deal_count_norm = (deal_count - deal_count.min()) / (deal_count.max() - deal_count.min())
    aum_norm = (aum - aum.min()) / (aum.max() - aum.min())
    combined = 0.5 * deal_count_norm + 0.5 * aum_norm
    return 10 + combined * 50

# Définition des colonnes
ALL_METRICS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %","WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)", "WAS", "Annualized Eq Rt (%)",
    "Deal Count", "AUM"
]

PERCENTAGE_METRICS = [
    "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Annualized Default Rate (%)", "Junior OC cushion", "IDT Cushion", "Caa %",
    "CCC % (S&P)", "Bond %", "Second Lien %", "Cov-Lite %",
    "Avg Col Quality Test/ Trigger %", "% of collateral rated B3",
    "Price < 80%", "Price < 70%", "Annualized Eq Rt (%)", "MV NAV (Equity)",
    "Equity St. Deviation"
]

# Paramètres par défaut
DEFAULT_RISK_COLS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %","WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)"
]

DEFAULT_REWARD_COLS = ["WAS", "Annualized Eq Rt (%)"]
BASE_METRICS = DEFAULT_RISK_COLS + DEFAULT_REWARD_COLS + ["Deal Count", "AUM"]

DEFAULT_RISK_INVERTS = {
    "WARF": False, "Caa/CCC Calculated %": False, "Defaulted %": False,
    "Largest industry concentration %": False, "Diversity": True,
    "Annualized Default Rate (%)": False, "Equity St. Deviation": False,
    "Junior OC cushion": True, "IDT Cushion": True, "Caa %": False,
    "CCC % (S&P)": False, "Bond %": False, "Cov-Lite %": False,
    "WA loans Price": True, "Avg Col Quality Test/ Trigger %": True,
    "% of collateral rated B3": False, "MV NAV (Equity)": True,"WAS/WARF": True
}

DEFAULT_REWARD_INVERTS = {col: False for col in DEFAULT_REWARD_COLS}

def load_data():
    st.sidebar.header("📤 Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Required format: CSV with ';' separator and ',' decimals"
    )

    if uploaded_files:
        all_frames = []
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file, sep=";", decimal=",", thousands=' ')

                # Extract date components from filename (MM-DD-YY)
                date_match = re.search(r"(\\d{2})-(\\d{2})-(\\d{2})", uploaded_file.name)
                if date_match:
                    m, d, y = date_match.groups()
                    date_val = pd.to_datetime(f"{m}-{d}-{y}", format="%m-%d-%y")
                    df["Year"] = date_val.year
                    df["Month"] = date_val.month
                    df["Date"] = date_val
                else:
                    df["Year"] = 0
                    df["Month"] = 0
                    df["Date"] = pd.NaT

                # Data cleaning
                df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
                for col in df.columns:
                    if col not in ["Manager Name", "Year", "Month", "Date"]:
                        df[col] = df[col].apply(clean_numeric)

                all_frames.append(df)
            except Exception as e:
                st.sidebar.error(f"Loading error for {uploaded_file.name}: {str(e)}")
                return None

        df_all = pd.concat(all_frames, ignore_index=True)
        st.session_state.df_raw = df_all.copy()
        st.session_state.df_clean = df_all
        st.session_state.file_uploaded = True
        st.session_state.hidden_managers = set()
        st.session_state.excluded_managers = set()
        st.sidebar.success("Files loaded successfully!")
        return df_all
    return None

def get_current_metric_types():
    """Return metrics classified by type according to current configuration"""
    risk_cols = st.session_state.get('custom_risk_cols', DEFAULT_RISK_COLS.copy())
    reward_cols = st.session_state.get('custom_reward_cols', DEFAULT_REWARD_COLS.copy())
    return risk_cols, reward_cols

def calculate_scores(df, risk_weights, reward_weights):
    # Get current metric classification
    risk_cols, reward_cols = get_current_metric_types()

    # Filter out excluded managers
    working_df = df[~df["Manager Name"].isin(st.session_state.get("excluded_managers", set()))]

    # Risk Score
    risk_score_numerator = 0
    risk_score_denominator = 0

    for col in risk_cols:
        scaled_col = f"Scaled_{col}"
        if scaled_col in working_df.columns:
            weight = risk_weights.get(col, 0)
            risk_score_numerator += working_df[scaled_col] * weight
            risk_score_denominator += weight

    working_df["Risk_Score"] = (risk_score_numerator / risk_score_denominator) if risk_score_denominator > 0 else 0.5

    # Reward Score
    reward_score_numerator = 0
    reward_score_denominator = 0

    for col in reward_cols:
        scaled_col = f"Scaled_{col}"
        if scaled_col in working_df.columns:
            weight = reward_weights.get(col, 0)
            reward_score_numerator += working_df[scaled_col] * weight
            reward_score_denominator += weight

    working_df["Reward_Score"] = (reward_score_numerator / reward_score_denominator) if reward_score_denominator > 0 else 0.5

    working_df["Average_Score"] = ((1-working_df["Risk_Score"]) + working_df["Reward_Score"]) / 2

    if "Deal Count" in working_df.columns and "AUM" in working_df.columns:
        working_df["Bubble_Size"] = calculate_bubble_size(working_df["Deal Count"], working_df["AUM"])
    else:
        working_df["Bubble_Size"] = 30

    # Merge back with original dataframe
    merge_cols = ["Manager Name"]
    if "Date" in df.columns:
        merge_cols.append("Date")
    elif "Year" in df.columns:
        merge_cols.append("Year")
    df = df.merge(
        working_df[merge_cols + ["Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]],
        on=merge_cols,
        how="left",
        suffixes=('', '_y')
    )

    return df

def metric_type_editor():
    st.sidebar.header("🔀 Metric Type Configuration")

    risk_cols, reward_cols = get_current_metric_types()
    available_metrics = [m for m in ALL_METRICS if m not in risk_cols and m not in reward_cols]

    with st.sidebar.expander("Change Metric Types"):
        st.write("**Current Risk Metrics:**")
        for metric in risk_cols:
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(metric)
            with cols[1]:
                if st.button("➡️ Reward", key=f"risk_to_reward_{metric}"):
                    reward_cols.append(metric)
                    risk_cols.remove(metric)
                    st.session_state.custom_risk_cols = risk_cols
                    st.session_state.custom_reward_cols = reward_cols
                    st.rerun()

        st.write("**Current Reward Metrics:**")
        for metric in reward_cols:
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(metric)
            with cols[1]:
                if st.button("⬅️ Risk", key=f"reward_to_risk_{metric}"):
                    risk_cols.append(metric)
                    reward_cols.remove(metric)
                    st.session_state.custom_risk_cols = risk_cols
                    st.session_state.custom_reward_cols = reward_cols
                    st.rerun()

        if available_metrics:
            st.write("**Available Metrics:**")
            for metric in available_metrics:
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write(metric)
                with cols[1]:
                    if st.button("➕Risk", key=f"add_risk_{metric}"):
                        risk_cols.append(metric)
                        st.session_state.custom_risk_cols = risk_cols
                        st.rerun()
                with cols[2]:
                    if st.button("➕Reward", key=f"add_reward_{metric}"):
                        reward_cols.append(metric)
                        st.session_state.custom_reward_cols = reward_cols
                        st.rerun()

def metric_editor(df):
    st.header("✏️ Manager Metrics Editor")

    if 'df_clean' not in st.session_state:
        st.warning("Please upload a file first")
        return

    if 'editable_df' not in st.session_state:
        st.session_state.editable_df = st.session_state.df_clean.copy()

    selected_manager = st.selectbox(
        "Select a manager",
        st.session_state.editable_df["Manager Name"].unique(),
        key="manager_select",
    )

    manager_idx = st.session_state.editable_df[
        st.session_state.editable_df["Manager Name"] == selected_manager
    ].index[0]

    st.markdown(f"### Editing: **{selected_manager}**")

    cols = st.columns(3)
    current_col = 0
    modifications = {}

    for col in BASE_METRICS:
        if col in st.session_state.editable_df.columns():
            ...
