# app.py
import io
import os
import re
import time
import locale
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from locale import setlocale, LC_NUMERIC

# ==============================
# Basic locale setup (safe)
# ==============================
try:
    setlocale(LC_NUMERIC, 'en_US.UTF-8')
except Exception:
    try:
        setlocale(LC_NUMERIC, '')
    except Exception:
        pass

# ==============================
# Streamlit page config & styles
# ==============================
st.set_page_config(
    layout="wide",
    page_title="Manager Risk Analysis",
    page_icon="📈",
)

st.markdown("""
    <style>
        .stApp { font-family: 'Helvetica Neue', Arial, sans-serif; color: #333333; }
        h1, h2, h3 {
            color: #2c3e50; font-weight: 600; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.3em;
        }
        .metric-container {
            background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px;
            padding: 1.5em; margin-bottom: 1.5em;
        }
        .stButton>button {
            background-color: #2c3e50; color: white; border-radius: 4px; border: none;
            padding: 0.5em 1em; font-weight: 500;
        }
        .stButton>button:hover { background-color: #1a2634; }
        .manager-list {
            max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0;
            padding: 10px; border-radius: 4px;
        }
        .metric-type-btn { font-size: 0.8em; padding: 0.2em 0.5em; margin: 0.1em; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 1em;'>Manager Risk-Reward Analysis</h1>", unsafe_allow_html=True)

# ==============================
# Helpers — cleaning & scaling
# ==============================
def clean_european_number(value):
    if pd.isna(value) or value == '': 
        return 0.0
    if isinstance(value, str):
        # remove spaces as thousand sep, remove dots as thousand sep, then use comma as decimal
        value = value.strip().replace(' ', '').replace('.', '').replace(',', '.')
    try:
        return float(value)
    except Exception:
        return 0.0

def clean_numeric(value, is_manager_name=False):
    if is_manager_name:
        return str(value).strip()
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, str):
        v = value.strip().strip('"').replace(' ', '')  # remove thousand spaces
        if '%' in v:
            try:
                return clean_european_number(v.replace('%', '')) / 100
            except Exception:
                return 0.0
        # Try EU-first, then fallback to plain float
        try:
            return clean_european_number(v)
        except Exception:
            try:
                return float(v)
            except Exception:
                return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0

def safe_min_max_scale(series, invert=False):
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if invert:
        s = (mx - s) + mn
    if mx == mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return ((s - mn) / (mx - mn)).round(6)

def calculate_bubble_size(deal_count, aum):
    d = pd.to_numeric(deal_count, errors='coerce').fillna(0.0)
    a = pd.to_numeric(aum, errors='coerce').fillna(0.0)
    dmn, dmx = float(d.min()), float(d.max())
    amn, amx = float(a.min()), float(a.max())
    d_norm = (d - dmn) / (dmx - dmn) if dmx != dmn else pd.Series(0.5, index=d.index)
    a_norm = (a - amn) / (amx - amn) if amx != amn else pd.Series(0.5, index=a.index)
    combined = 0.5 * d_norm + 0.5 * a_norm
    return 10 + combined * 50

# ==============================
# Metrics / defaults
# ==============================
ALL_METRICS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %", "WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)", "WAS", "Annualized Eq Rt (%)",
    "Deal Count", "AUM"
]

PERCENTAGE_METRICS = {
    "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Annualized Default Rate (%)", "Junior OC cushion", "IDT Cushion", "Caa %",
    "CCC % (S&P)", "Bond %", "Second Lien %", "Cov-Lite %",
    "Avg Col Quality Test/ Trigger %", "% of collateral rated B3",
    "Price < 80%", "Price < 70%", "Annualized Eq Rt (%)", "MV NAV (Equity)",
    "Equity St. Deviation"
}

DEFAULT_RISK_COLS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %", "WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)"
]
DEFAULT_REWARD_COLS = ["WAS", "Annualized Eq Rt (%)"]

DEFAULT_RISK_INVERTS = {
    "WARF": False, "Caa/CCC Calculated %": False, "Defaulted %": False,
    "Largest industry concentration %": False, "Diversity": True,
    "Annualized Default Rate (%)": False, "Equity St. Deviation": False,
    "Junior OC cushion": True, "IDT Cushion": True, "Caa %": False,
    "CCC % (S&P)": False, "Bond %": False, "Cov-Lite %": False,
    "WA loans Price": True, "Avg Col Quality Test/ Trigger %": True,
    "% of collateral rated B3": False, "MV NAV (Equity)": True, "WAS/WARF": True
}
DEFAULT_REWARD_INVERTS = {c: False for c in DEFAULT_REWARD_COLS}

BASE_METRICS = DEFAULT_RISK_COLS + DEFAULT_REWARD_COLS + ["Deal Count", "AUM"]

# ==============================
# Robust CSV Reader
# ==============================
HEADER_PATTERNS = [
    r'^\s*Manager\s*Name\s*[\;,]',     # "Manager Name;" or "Manager Name,"
    r'^\s*Manager\s*Name\s*$'          # sometimes no delimiter at the end
]

def _detect_header_index_and_sep(text):
    lines = text.splitlines()
    # Try to find a line that looks like the header
    header_idx = None
    for i, line in enumerate(lines):
        for pat in HEADER_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                header_idx = i
                break
        if header_idx is not None:
            break
    if header_idx is None:
        # fallback: search for line that contains "Manager Name" and has many delimiters
        candidates = []
        for i, line in enumerate(lines):
            if "Manager Name" in line:
                semi = line.count(';')
                comma = line.count(',')
                candidates.append((i, semi, comma))
        if candidates:
            # pick the one with the most delimiters
            header_idx = sorted(candidates, key=lambda x: max(x[1], x[2]))[0][0]

    if header_idx is None:
        raise ValueError("Could not locate the header row containing 'Manager Name'.")

    header_line = lines[header_idx]
    semi, comma = header_line.count(';'), header_line.count(',')
    sep = ';' if semi >= comma else ','
    return header_idx, sep

def _read_manager_csv(uploaded_file):
    # Read bytes once
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    # Decode with utf-8 then fallback to latin-1
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    header_idx, sep = _detect_header_index_and_sep(text)

    # decimal: usual pairing is sep=';' with decimal=',', else '.' if comma-separated
    decimal = ',' if sep == ';' else '.'
    thousands = ' '  # you were using a space as thousands

    sio = io.StringIO(text)
    df = pd.read_csv(
        sio,
        sep=sep,
        header=header_idx,
        decimal=decimal,
        engine="python"
    )

    # Normalize columns
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

    # Drop fully empty cols that sometimes appear after messy exports
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Remove footer rows where Manager Name is empty
    if "Manager Name" in df.columns:
        df = df[~df["Manager Name"].isna() & (df["Manager Name"].astype(str).str.strip() != "")]

    # Reset index after filtering
    return df.reset_index(drop=True)

# ==============================
# Session getters
# ==============================
def get_current_metric_types():
    risk_cols = st.session_state.get('custom_risk_cols', DEFAULT_RISK_COLS.copy())
    reward_cols = st.session_state.get('custom_reward_cols', DEFAULT_REWARD_COLS.copy())
    return risk_cols, reward_cols

# ==============================
# Core calculations
# ==============================
def calculate_scores(df, risk_weights, reward_weights):
    # Work on a copy
    df = df.copy()

    # Current config
    risk_cols, reward_cols = get_current_metric_types()

    # Filter out excluded managers
    excluded = st.session_state.get("excluded_managers", set())
    working_df = df[~df["Manager Name"].isin(excluded)].copy()

    # Risk score
    risk_num = 0
    risk_den = 0
    for col in risk_cols:
        scaled = f"Scaled_{col}"
        if scaled in working_df.columns:
            w = float(risk_weights.get(col, 0.0))
            risk_num = risk_num + working_df[scaled] * w
            risk_den += w
    working_df["Risk_Score"] = (risk_num / risk_den) if risk_den > 0 else 0.5

    # Reward score
    rew_num = 0
    rew_den = 0
    for col in reward_cols:
        scaled = f"Scaled_{col}"
        if scaled in working_df.columns:
            w = float(reward_weights.get(col, 0.0))
            rew_num = rew_num + working_df[scaled] * w
            rew_den += w
    working_df["Reward_Score"] = (rew_num / rew_den) if rew_den > 0 else 0.5

    # Average score
    working_df["Average_Score"] = ((1 - working_df["Risk_Score"]) + working_df["Reward_Score"]) / 2

    # Bubble sizes
    if "Deal Count" in working_df.columns and "AUM" in working_df.columns:
        working_df["Bubble_Size"] = calculate_bubble_size(working_df["Deal Count"], working_df["AUM"])
    else:
        working_df["Bubble_Size"] = 30

    # Merge scores back
    on_cols = ["Manager Name"] + (["Year"] if "Year" in df.columns else [])
    df = df.merge(
        working_df[on_cols + ["Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]],
        on=on_cols,
        how="left",
        suffixes=('', '_y')
    )
    return df

# ==============================
# UI blocks
# ==============================
def metric_type_editor():
    st.sidebar.header("🔀 Metric Type Configuration")
    risk_cols, reward_cols = get_current_metric_types()
    # limit available metrics to those present in df
    present_cols = set(st.session_state.df_clean.columns)
    candidates = [m for m in ALL_METRICS if m in present_cols and m not in risk_cols and m not in reward_cols]

    with st.sidebar.expander("Change Metric Types"):
        st.write("**Current Risk Metrics:**")
        for metric in list(risk_cols):
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(metric)
            with cols[1]:
                if st.button("➡️ Reward", key=f"risk_to_reward_{metric}"):
                    risk_cols.remove(metric)
                    reward_cols.append(metric)
                    st.session_state.custom_risk_cols = risk_cols
                    st.session_state.custom_reward_cols = reward_cols
                    st.rerun()

        st.write("**Current Reward Metrics:**")
        for metric in list(reward_cols):
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(metric)
            with cols[1]:
                if st.button("⬅️ Risk", key=f"reward_to_risk_{metric}"):
                    reward_cols.remove(metric)
                    risk_cols.append(metric)
                    st.session_state.custom_risk_cols = risk_cols
                    st.session_state.custom_reward_cols = reward_cols
                    st.rerun()

        if candidates:
            st.write("**Available Metrics:**")
            for metric in candidates:
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

    managers = st.session_state.editable_df["Manager Name"].dropna().unique()
    if len(managers) == 0:
        st.warning("No managers found.")
        return

    selected_manager = st.selectbox("Select a manager", managers, key="manager_select")
    manager_idx = st.session_state.editable_df[
        st.session_state.editable_df["Manager Name"] == selected_manager
    ].index[0]

    st.markdown(f"### Editing: **{selected_manager}**")

    cols = st.columns(3)
    current_col = 0
    modifications = {}

    # Only edit metrics actually present
    for col in [c for c in BASE_METRICS if c in st.session_state.editable_df.columns]:
        with cols[current_col]:
            current_val = float(pd.to_numeric(st.session_state.editable_df.at[manager_idx, col], errors='coerce') or 0.0)
            if col in PERCENTAGE_METRICS:
                new_val = st.number_input(
                    f"{col} (%)", value=float(current_val * 100.0),
                    min_value=0.0, max_value=100.0, step=0.1, key=f"edit_{col}_{selected_manager}"
                )
                modifications[col] = new_val / 100.0
            else:
                new_val = st.number_input(
                    col, value=float(current_val), step=0.1, key=f"edit_{col}_{selected_manager}"
                )
                modifications[col] = new_val
        current_col = (current_col + 1) % 3

    if st.button("💾 Save", key=f"save_{selected_manager}"):
        try:
            # Apply modifications
            for c, v in modifications.items():
                st.session_state.editable_df.at[manager_idx, c] = v

            # Recompute scaled columns for modified ones
            risk_cols, reward_cols = get_current_metric_types()
            affected = set(modifications.keys()) & set(risk_cols + reward_cols)
            for col in affected:
                invert = st.session_state.get(
                    f"invert_{col}",
                    DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols else DEFAULT_REWARD_INVERTS.get(col, False)
                )
                if "Year" in st.session_state.editable_df.columns:
                    st.session_state.editable_df[f"Scaled_{col}"] = (
                        st.session_state.editable_df
                        .groupby("Year")[col]
                        .transform(lambda s: safe_min_max_scale(s, invert))
                    )
                else:
                    st.session_state.editable_df[f"Scaled_{col}"] = safe_min_max_scale(
                        st.session_state.editable_df[col], invert
                    )

            # Recalculate scores
            risk_weights = {c: st.session_state.get(f'risk_{c}', 1.0) for c in risk_cols}
            reward_weights = {c: st.session_state.get(f'reward_{c}', 1.0) for c in reward_cols}

            updated_df = calculate_scores(st.session_state.editable_df, risk_weights, reward_weights)

            # Update main data
            st.session_state.df_clean = updated_df.copy()
            st.session_state.editable_df = updated_df.copy()
            st.session_state.last_update = time.time()

            st.success("Changes saved successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def manager_selection_interface():
    st.header("👥 Manager Selection")

    if 'df_clean' not in st.session_state:
        st.warning("Please upload data first")
        return

    all_managers = st.session_state.df_clean["Manager Name"].dropna().unique()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hide from Visualization")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        hidden = st.session_state.get("hidden_managers", set())
        new_hidden = set()
        for m in all_managers:
            if st.checkbox(f"Hide {m}", value=(m in hidden), key=f"hide_{m}"):
                new_hidden.add(m)
        st.session_state.hidden_managers = new_hidden
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Exclude from Calculations")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        excluded = st.session_state.get("excluded_managers", set())
        new_excluded = set()
        for m in all_managers:
            if st.checkbox(f"Exclude {m}", value=(m in excluded), key=f"exclude_{m}"):
                new_excluded.add(m)
        st.session_state.excluded_managers = new_excluded
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Data loader (robust)
# ==============================
def load_data():
    st.sidebar.header("📤 Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="CSV may include preamble rows; the app will auto-detect the true header."
    )

    if not uploaded_files:
        return None

    all_frames = []
    for uploaded_file in uploaded_files:
        try:
            df = _read_manager_csv(uploaded_file)

            # Extract year from filename
            year_match = re.search(r"(19|20)\d{2}", uploaded_file.name)
            year = int(year_match.group()) if year_match else 0
            if "Year" not in df.columns:
                df["Year"] = year
            else:
                # only fill NA with detected year
                df["Year"] = pd.to_numeric(df["Year"], errors='coerce').fillna(year).astype(int)

            # Ensure Manager Name
            if "Manager Name" not in df.columns:
                raise ValueError("The CSV does not contain 'Manager Name' column after parsing.")

            # Clean fields
            df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
            numeric_cols = [c for c in df.columns if c not in ["Manager Name", "Year"]]
            for c in numeric_cols:
                df[c] = df[c].apply(clean_numeric)

            all_frames.append(df)
        except Exception as e:
            st.sidebar.error(f"Loading error for {uploaded_file.name}: {str(e)}")
            return None

    if not all_frames:
        return None

    df_all = pd.concat(all_frames, ignore_index=True)

    # Initialize session state
    st.session_state.df_raw = df_all.copy()
    st.session_state.df_clean = df_all.copy()
    st.session_state.file_uploaded = True
    st.session_state.hidden_managers = set()
    st.session_state.excluded_managers = set()
    st.sidebar.success("Files loaded successfully!")
    return df_all

# ==============================
# Main app
# ==============================
def main():
    # Init defaults once
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.custom_risk_cols = DEFAULT_RISK_COLS.copy()
        st.session_state.custom_reward_cols = DEFAULT_REWARD_COLS.copy()

    # Load data (once)
    if not st.session_state.file_uploaded:
        df = load_data()
        if df is None:
            st.info("Upload at least one CSV to get started.")
            return

    if 'df_clean' not in st.session_state or "Manager Name" not in st.session_state.df_clean.columns:
        st.warning("Please upload a valid CSV file containing a 'Manager Name' header.")
        return

    # Reset button
    if st.session_state.file_uploaded and st.sidebar.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

    # Inversion UI
    st.sidebar.header("🔄 Metric Inversion")
    risk_cols, reward_cols = get_current_metric_types()

    with st.sidebar.expander("Risk Metrics Inversion"):
        for col in risk_cols:
            if col in st.session_state.df_clean.columns:
                st.checkbox(
                    f"Invert {col}",
                    value=DEFAULT_RISK_INVERTS.get(col, False),
                    key=f"invert_{col}"
                )

    with st.sidebar.expander("Reward Metrics Inversion"):
        for col in reward_cols:
            if col in st.session_state.df_clean.columns:
                st.checkbox(
                    f"Invert {col}",
                    value=DEFAULT_REWARD_INVERTS.get(col, False),
                    key=f"invert_{col}"
                )

    # Metric type config
    metric_type_editor()

    # Weights
    st.sidebar.header("⚖️ Weight Configuration")
    risk_cols, reward_cols = get_current_metric_types()

    with st.sidebar.expander("Risk Weights"):
        risk_weights = {}
        for col in risk_cols:
            if col in st.session_state.df_clean.columns:
                key = f'risk_{col}'
                if key not in st.session_state:
                    st.session_state[key] = 1.0
                st.session_state[key] = st.slider(f"{col}", 0.0, 2.0, st.session_state[key], 0.1, key=f"risk_slider_{col}")
                risk_weights[col] = st.session_state[key]

    with st.sidebar.expander("Reward Weights"):
        reward_weights = {}
        for col in reward_cols:
            if col in st.session_state.df_clean.columns:
                key = f'reward_{col}'
                if key not in st.session_state:
                    st.session_state[key] = 1.0
                st.session_state[key] = st.slider(f"{col}", 0.0, 2.0, st.session_state[key], 0.1, key=f"reward_slider_{col}")
                reward_weights[col] = st.session_state[key]

    # Scale metrics with current inversion
    for col in list(set(risk_cols + reward_cols)):
        if col in st.session_state.df_clean.columns:
            invert = st.session_state.get(
                f"invert_{col}",
                DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols else DEFAULT_REWARD_INVERTS.get(col, False)
            )
            if "Year" in st.session_state.df_clean.columns:
                st.session_state.df_clean[f"Scaled_{col}"] = (
                    st.session_state.df_clean
                    .groupby("Year")[col]
                    .transform(lambda s: safe_min_max_scale(s, invert))
                )
            else:
                st.session_state.df_clean[f"Scaled_{col}"] = safe_min_max_scale(st.session_state.df_clean[col], invert)

    # Compute scores
    current_df = calculate_scores(st.session_state.df_clean.copy(), risk_weights, reward_weights)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "👥 Manager Selection", "✏️ Edit Metrics"])

    with tab1:
        st.header("Risk-Reward Matrix")
        hidden = st.session_state.get("hidden_managers", set())
        filtered_df = current_df[~current_df["Manager Name"].isin(hidden)].copy()

        # Highlight selection
        highlighted = st.multiselect("Highlight managers", filtered_df["Manager Name"].unique(), max_selections=4)
        filtered_df["Highlight"] = np.where(filtered_df["Manager Name"].isin(highlighted), filtered_df["Manager Name"], "Other")

        # Colors
        palette = px.colors.qualitative.Bold
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(highlighted)}
        color_map["Other"] = "#CCCCCC"

        # Plot args
        scatter_args = dict(
            x="Risk_Score", y="Reward_Score", size="Bubble_Size",
            color="Highlight", hover_name="Manager Name",
            hover_data={"Risk_Score": ":.3f", "Reward_Score": ":.3f", "Average_Score": ":.3f", "Deal Count": True, "AUM": ":,.0f"},
            size_max=40, color_discrete_map=color_map
        )
        if "Year" in filtered_df.columns:
            scatter_args["animation_frame"] = "Year"

        fig = px.scatter(filtered_df, **scatter_args)
        fig.update_layout(
            xaxis_range=[-0.1, 1.1], yaxis_range=[-0.1, 1.1],
            height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333333"), margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Metrics Table", expanded=False):
            table_df = filtered_df.set_index(["Year", "Manager Name"]) if "Year" in filtered_df.columns else filtered_df.set_index("Manager Name")
            st.dataframe(table_df, use_container_width=True)

    with tab2:
        manager_selection_interface()

    with tab3:
        metric_editor(current_df)

    # Export
    st.sidebar.header("📤 Export Options")
    if st.sidebar.button("💾 Generate Report"):
        with st.spinner("Generating report..."):
            try:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)

                # Metric classification snapshot
                risk_cols, reward_cols = get_current_metric_types()

                export_cols = [
                    "Manager Name", "Year", "Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"
                ] + [c for c in risk_cols if c in current_df.columns] \
                  + [c for c in reward_cols if c in current_df.columns] \
                  + [c for c in ["Deal Count", "AUM"] if c in current_df.columns]

                export_df = current_df[[c for c in export_cols if c in current_df.columns]].copy()

                # CSV
                csv_filename = f"manager_scores_{timestamp}.csv"
                csv_path = f"output/{csv_filename}"
                export_df.to_csv(csv_path, index=False, sep=";", decimal=",")

                # Excel
                excel_filename = f"manager_scores_{timestamp}.xlsx"
                excel_path = f"output/{excel_filename}"

                with pd.ExcelWriter(excel_path) as writer:
                    export_df.to_excel(writer, sheet_name="Scores", index=False)

                    selections_df = pd.DataFrame({
                        "Manager": current_df["Manager Name"],
                        "Hidden": current_df["Manager Name"].isin(st.session_state.get("hidden_managers", set())),
                        "Excluded": current_df["Manager Name"].isin(st.session_state.get("excluded_managers", set()))
                    })
                    selections_df.to_excel(writer, sheet_name="Manager Selections", index=False)

                    metric_types_df = pd.DataFrame({
                        "Metric": [*risk_cols, *reward_cols],
                        "Type": ["Risk"]*len(risk_cols) + ["Reward"]*len(reward_cols),
                        "Default Inverted": [
                            DEFAULT_RISK_INVERTS.get(c, False) if c in DEFAULT_RISK_COLS else DEFAULT_REWARD_INVERTS.get(c, False)
                            for c in [*risk_cols, *reward_cols]
                        ]
                    })
                    metric_types_df.to_excel(writer, sheet_name="Metric Types", index=False)

                    inversion_df = pd.DataFrame([
                        {
                            "Metric": c,
                            "Inverted": st.session_state.get(
                                f"invert_{c}",
                                DEFAULT_RISK_INVERTS.get(c, False) if c in risk_cols else DEFAULT_REWARD_INVERTS.get(c, False)
                            ),
                            "Type": "Risk" if c in risk_cols else "Reward"
                        }
                        for c in [*risk_cols, *reward_cols]
                    ])
                    inversion_df.to_excel(writer, sheet_name="Inversion Settings", index=False)

                    weights_df = pd.DataFrame({
                        "Metric": [*risk_cols, *reward_cols],
                        "Weight": [st.session_state.get(f'risk_{c}', 1.0) for c in risk_cols] +
                                  [st.session_state.get(f'reward_{c}', 1.0) for c in reward_cols],
                        "Type": ["Risk"]*len(risk_cols) + ["Reward"]*len(reward_cols)
                    })
                    weights_df.to_excel(writer, sheet_name="Weights", index=False)

                st.sidebar.success("Report generated successfully!")

                col1, col2 = st.sidebar.columns(2)
                with col1:
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 CSV", f, file_name=csv_filename, mime="text/csv")
                with col2:
                    with open(excel_path, "rb") as f:
                        st.download_button("📊 Excel", f, file_name=excel_filename,
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.sidebar.error(f"Export error: {str(e)}")

if __name__ == "__main__":
    main()
