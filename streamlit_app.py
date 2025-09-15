# app.py
import io
import os
import re
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from locale import setlocale, LC_NUMERIC

# ───────── Page config & styles ─────────
try:
    setlocale(LC_NUMERIC, 'en_US.UTF-8')
except Exception:
    try: setlocale(LC_NUMERIC, '')
    except Exception: pass

st.set_page_config(layout="wide", page_title="Manager Risk Analysis", page_icon="📈")
st.markdown("""
<style>
  .stApp { font-family: 'Helvetica Neue', Arial, sans-serif; color:#333; }
  h1,h2,h3 { color:#2c3e50; font-weight:600; border-bottom:1px solid #e0e0e0; padding-bottom:.3em; }
  .manager-list { max-height: 400px; overflow-y:auto; border:1px solid #e0e0e0; padding:10px; border-radius:4px; }
  .stButton>button { background:#2c3e50; color:#fff; border:none; border-radius:4px; padding:.5em 1em; }
  .stButton>button:hover { background:#1a2634; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;margin-bottom:1em;'>Manager Risk-Reward Analysis</h1>", unsafe_allow_html=True)

# ───────── Constants / defaults ─────────
ALL_METRICS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %", "WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)", "WAS", "Annualized Eq Rt (%)",
    "Deal Count", "AUM", "Price < 80%", "Price < 70%"
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
    "% of collateral rated B3", "MV NAV (Equity)", "Price < 80%", "Price < 70%"
]
DEFAULT_REWARD_COLS = ["WAS", "Annualized Eq Rt (%)"]

DEFAULT_RISK_INVERTS = {
    "WARF": False, "Caa/CCC Calculated %": False, "Defaulted %": False,
    "Largest industry concentration %": False, "Diversity": True,
    "Annualized Default Rate (%)": False, "Equity St. Deviation": False,
    "Junior OC cushion": True, "IDT Cushion": True, "Caa %": False,
    "CCC % (S&P)": False, "Bond %": False, "Cov-Lite %": False,
    "WA loans Price": True, "Avg Col Quality Test/ Trigger %": True,
    "% of collateral rated B3": False, "MV NAV (Equity)": True, "WAS/WARF": True,
    "Price < 80%": False, "Price < 70%": False
}
DEFAULT_REWARD_INVERTS = {c: False for c in DEFAULT_REWARD_COLS}

BASE_METRICS = DEFAULT_RISK_COLS + DEFAULT_REWARD_COLS + ["Deal Count", "AUM"]

TARGET_STOP_COL = "Price < 70%"  # ← we will trim columns to this (inclusive)

# ───────── CSV reading (semicolon only) ─────────
def _detect_header_index_semicolon(text: str) -> int:
    lines = text.splitlines()
    # Prefer exact "Manager Name;" line
    for i, line in enumerate(lines):
        if re.search(r'^\s*Manager\s*Name\s*;', line, flags=re.IGNORECASE):
            return i
    # Fallback: line with "Manager Name" and MOST semicolons
    cands = [(i, line.count(';')) for i, line in enumerate(lines) if "Manager Name" in line]
    if cands:
        return sorted(cands, key=lambda x: x[1], reverse=True)[0][0]
    # Last resort: first line with our TARGET_STOP_COL
    for i, line in enumerate(lines):
        if TARGET_STOP_COL in line and line.count(';') >= 1:
            return i
    raise ValueError("Could not locate a header row (no 'Manager Name;' or target column).")

def _dedup_columns(cols):
    seen = {}
    uniq = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}.{seen[c]}")
    return uniq

def _trim_until_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Keep columns from the left up to and including 'target'.
    If duplicates exist (e.g., 'Price < 70%', 'Price < 70%.1'), we use the first match.
    """
    # find first exact match or a deduped variant starting with target
    target_candidates = [c for c in df.columns if c == target or c.startswith(f"{target}.")]
    if not target_candidates:
        # if not present, we don't trim (we keep all columns)
        return df
    target_col = target_candidates[0]
    idx = list(df.columns).index(target_col)
    keep_cols = list(df.columns)[: idx + 1]
    return df.loc[:, keep_cols]

def _read_semicolon_manager_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    header_idx = _detect_header_index_semicolon(text)

    # Parse strict semicolon CSV with EU decimal comma
    sio = io.StringIO(text)
    df = pd.read_csv(sio, sep=';', header=header_idx, decimal=',', engine="python")

    # Normalize headers: collapse spaces and dedup
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    df.columns = _dedup_columns(df.columns)

    # Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Trim to the left until (and including) TARGET_STOP_COL
    df = _trim_until_target(df, TARGET_STOP_COL)

    # Keep only rows with a non-empty Manager Name (choose first matching col name)
    mcols = [c for c in df.columns if c == "Manager Name" or c.startswith("Manager Name.")]
    if mcols:
        mcol = mcols[0]
        df = df[~df[mcol].isna() & (df[mcol].astype(str).str.strip() != "")]
        if mcol != "Manager Name":
            df = df.rename(columns={mcol: "Manager Name"})
    # If no Manager Name present, we still proceed with trimmed df.

    return df.reset_index(drop=True)

# ───────── Cleaning & scaling helpers ─────────
def clean_european_number(value):
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, str):
        value = value.strip().replace(' ', '').replace('.', '').replace(',', '.')
    try:
        return float(value)
    except Exception:
        return 0.0

def clean_numeric(value, is_manager_name=False):
    import numpy as np
    if isinstance(value, (pd.Series, list, tuple, np.ndarray)):
        s = pd.Series(value).dropna()
        value = s.iloc[0] if not s.empty else np.nan

    if is_manager_name:
        return "" if pd.isna(value) else str(value).strip()

    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        v = value.strip().strip('"').replace(' ', '')
        if v == "":
            return 0.0
        if '%' in v:
            num = v.replace('%', '').replace('.', '').replace(',', '.')
            try: return float(num) / 100.0
            except Exception: return 0.0
        v2 = v.replace('.', '').replace(',', '.')
        try: return float(v2)
        except Exception: return 0.0
    try: return float(value)
    except Exception: return 0.0

def safe_min_max_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if invert: s = (mx - s) + mn
    if mx == mn: return pd.Series(np.full(len(s), 0.5), index=s.index)
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

def get_current_metric_types():
    risk_cols = st.session_state.get('custom_risk_cols', DEFAULT_RISK_COLS.copy())
    reward_cols = st.session_state.get('custom_reward_cols', DEFAULT_REWARD_COLS.copy())
    return risk_cols, reward_cols

# ───────── Scoring ─────────
def calculate_scores(df: pd.DataFrame, risk_weights: dict, reward_weights: dict) -> pd.DataFrame:
    df = df.copy()
    risk_cols, reward_cols = get_current_metric_types()
    excluded = st.session_state.get("excluded_managers", set())
    working_df = df if "Manager Name" not in df.columns else df[~df["Manager Name"].isin(excluded)].copy()

    # Risk
    risk_num, risk_den = 0, 0.0
    for col in risk_cols:
        sc = f"Scaled_{col}"
        if sc in working_df.columns:
            w = float(risk_weights.get(col, 0.0))
            risk_num = risk_num + working_df[sc] * w
            risk_den += w
    working_df["Risk_Score"] = (risk_num / risk_den) if risk_den > 0 else 0.5

    # Reward
    rew_num, rew_den = 0, 0.0
    for col in reward_cols:
        sc = f"Scaled_{col}"
        if sc in working_df.columns:
            w = float(reward_weights.get(col, 0.0))
            rew_num = rew_num + working_df[sc] * w
            rew_den += w
    working_df["Reward_Score"] = (rew_num / rew_den) if rew_den > 0 else 0.5

    working_df["Average_Score"] = ((1 - working_df["Risk_Score"]) + working_df["Reward_Score"]) / 2

    if "Deal Count" in working_df.columns and "AUM" in working_df.columns:
        working_df["Bubble_Size"] = calculate_bubble_size(working_df["Deal Count"], working_df["AUM"])
    else:
        working_df["Bubble_Size"] = 30

    on_cols = (["Manager Name"] if "Manager Name" in df.columns else []) + (["Year"] if "Year" in df.columns else [])
    if on_cols:
        df = df.merge(
            working_df[on_cols + ["Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]],
            on=on_cols, how="left"
        )
    else:
        df[["Risk_Score","Reward_Score","Average_Score","Bubble_Size"]] = working_df[["Risk_Score","Reward_Score","Average_Score","Bubble_Size"]]
    return df

# ───────── UI blocks ─────────
def metric_type_editor():
    st.sidebar.header("🔀 Metric Type Configuration")
    risk_cols, reward_cols = get_current_metric_types()
    present_cols = set(st.session_state.df_clean.columns)
    candidates = [m for m in ALL_METRICS if m in present_cols and m not in risk_cols and m not in reward_cols]

    with st.sidebar.expander("Change Metric Types"):
        st.write("**Current Risk Metrics:**")
        for metric in list(risk_cols):
            cols = st.columns([4, 1]); cols[0].write(metric)
            if cols[1].button("➡️ Reward", key=f"risk_to_reward_{metric}"):
                risk_cols.remove(metric); reward_cols.append(metric)
                st.session_state.custom_risk_cols = risk_cols; st.session_state.custom_reward_cols = reward_cols
                st.rerun()
        st.write("**Current Reward Metrics:**")
        for metric in list(reward_cols):
            cols = st.columns([4, 1]); cols[0].write(metric)
            if cols[1].button("⬅️ Risk", key=f"reward_to_risk_{metric}"):
                reward_cols.remove(metric); risk_cols.append(metric)
                st.session_state.custom_risk_cols = risk_cols; st.session_state.custom_reward_cols = reward_cols
                st.rerun()
        if candidates:
            st.write("**Available Metrics:**")
            for metric in candidates:
                cols = st.columns([3, 1, 1]); cols[0].write(metric)
                if cols[1].button("➕Risk", key=f"add_risk_{metric}"):
                    risk_cols.append(metric); st.session_state.custom_risk_cols = risk_cols; st.rerun()
                if cols[2].button("➕Reward", key=f"add_reward_{metric}"):
                    reward_cols.append(metric); st.session_state.custom_reward_cols = reward_cols; st.rerun()

def metric_editor(df):
    st.header("✏️ Manager Metrics Editor")
    if 'df_clean' not in st.session_state: st.warning("Please upload a file first"); return
    if 'editable_df' not in st.session_state: st.session_state.editable_df = st.session_state.df_clean.copy()
    if "Manager Name" not in st.session_state.editable_df.columns:
        st.info("No 'Manager Name' column found. Editing disabled for this file."); return

    managers = st.session_state.editable_df["Manager Name"].dropna().unique()
    if len(managers) == 0: st.warning("No managers found."); return

    selected_manager = st.selectbox("Select a manager", managers, key="manager_select")
    manager_idx = st.session_state.editable_df[st.session_state.editable_df["Manager Name"] == selected_manager].index[0]
    st.markdown(f"### Editing: **{selected_manager}**")

    cols = st.columns(3); current_col = 0; modifications = {}
    for col in [c for c in BASE_METRICS if c in st.session_state.editable_df.columns]:
        with cols[current_col]:
            raw = st.session_state.editable_df.at[manager_idx, col]
            num = pd.to_numeric(raw, errors='coerce'); current_val = 0.0 if pd.isna(num) else float(num)
            if col in PERCENTAGE_METRICS:
                new_val = st.number_input(f"{col} (%)", value=float(current_val * 100.0), min_value=0.0, max_value=100.0, step=0.1, key=f"edit_{col}_{selected_manager}")
                modifications[col] = new_val / 100.0
            else:
                new_val = st.number_input(col, value=float(current_val), step=0.1, key=f"edit_{col}_{selected_manager}")
                modifications[col] = new_val
        current_col = (current_col + 1) % 3

    if st.button("💾 Save", key=f"save_{selected_manager}"):
        try:
            for c, v in modifications.items():
                st.session_state.editable_df.at[manager_idx, c] = v
            risk_cols, reward_cols = get_current_metric_types()
            affected = set(modifications.keys()) & set(risk_cols + reward_cols)
            for col in affected:
                invert = st.session_state.get(f"invert_{col}", DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols else DEFAULT_REWARD_INVERTS.get(col, False))
                if "Year" in st.session_state.editable_df.columns:
                    st.session_state.editable_df[f"Scaled_{col}"] = st.session_state.editable_df.groupby("Year")[col].transform(lambda s: safe_min_max_scale(s, invert))
                else:
                    st.session_state.editable_df[f"Scaled_{col}"] = safe_min_max_scale(st.session_state.editable_df[col], invert)
            risk_weights = {c: st.session_state.get(f'risk_{c}', 1.0) for c in risk_cols}
            reward_weights = {c: st.session_state.get(f'reward_{c}', 1.0) for c in reward_cols}
            updated = calculate_scores(st.session_state.editable_df, risk_weights, reward_weights)
            st.session_state.df_clean = updated.copy(); st.session_state.editable_df = updated.copy(); st.session_state.last_update = time.time()
            st.success("Changes saved successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def manager_selection_interface():
    st.header("👥 Manager Selection")
    if 'df_clean' not in st.session_state: st.warning("Please upload data first"); return
    if "Manager Name" not in st.session_state.df_clean.columns:
        st.info("No 'Manager Name' column found. Selection disabled for this file."); return

    all_managers = st.session_state.df_clean["Manager Name"].dropna().unique()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hide from Visualization")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        hidden = st.session_state.get("hidden_managers", set()); new_hidden = set()
        for m in all_managers:
            if st.checkbox(f"Hide {m}", value=(m in hidden), key=f"hide_{m}"): new_hidden.add(m)
        st.session_state.hidden_managers = new_hidden
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.subheader("Exclude from Calculations")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        excluded = st.session_state.get("excluded_managers", set()); new_excluded = set()
        for m in all_managers:
            if st.checkbox(f"Exclude {m}", value=(m in excluded), key=f"exclude_{m}"): new_excluded.add(m)
        st.session_state.excluded_managers = new_excluded
        st.markdown("</div>", unsafe_allow_html=True)

# ───────── Data loading (semicolon CSVs only, trim at TARGET_STOP_COL) ─────────
def load_data():
    st.sidebar.header("📤 Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        f"Upload your semicolon (;) CSV files — columns to the right of '{TARGET_STOP_COL}' are ignored",
        type=["csv"], accept_multiple_files=True,
        help=f"Preamble rows allowed. The app auto-detects the true header and keeps columns only up to '{TARGET_STOP_COL}'."
    )
    if not uploaded_files: return None

    frames = []
    for uploaded_file in uploaded_files:
        try:
            df = _read_semicolon_manager_csv(uploaded_file)

            # Year from filename
            year_match = re.search(r"(19|20)\d{2}", uploaded_file.name)
            year = int(year_match.group()) if year_match else 0
            if "Year" not in df.columns:
                df["Year"] = year
            else:
                df["Year"] = pd.to_numeric(df["Year"], errors='coerce').fillna(year).astype(int)

            # Cleaning
            if "Manager Name" in df.columns:
                df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
            for c in [c for c in df.columns if c not in ["Manager Name", "Year"]]:
                df[c] = df[c].apply(clean_numeric)

            frames.append(df)
        except Exception as e:
            st.sidebar.error(f"Loading error for {uploaded_file.name}: {str(e)}")
            return None

    if not frames: return None

    df_all = pd.concat(frames, ignore_index=True)
    st.session_state.df_raw = df_all.copy()
    st.session_state.df_clean = df_all.copy()
    st.session_state.file_uploaded = True
    st.session_state.hidden_managers = set()
    st.session_state.excluded_managers = set()
    st.sidebar.success("Files loaded successfully! (Columns after '%s' were ignored.)" % TARGET_STOP_COL)
    return df_all

# ───────── Main app ─────────
def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.custom_risk_cols = DEFAULT_RISK_COLS.copy()
        st.session_state.custom_reward_cols = DEFAULT_REWARD_COLS.copy()

    if not st.session_state.file_uploaded:
        df = load_data()
        if df is None:
            st.info(f"Upload at least one semicolon CSV (columns after '{TARGET_STOP_COL}' will be ignored).")
            return

    if 'df_clean' not in st.session_state:
        st.warning("Please upload a valid semicolon CSV."); return

    # Reset
    if st.session_state.file_uploaded and st.sidebar.button("🔄 Reset"):
        st.session_state.clear(); st.rerun()

    # Inversion controls
    st.sidebar.header("🔄 Metric Inversion")
    risk_cols, reward_cols = get_current_metric_types()
    with st.sidebar.expander("Risk Metrics Inversion"):
        for col in risk_cols:
            if col in st.session_state.df_clean.columns:
                st.checkbox(f"Invert {col}", value=DEFAULT_RISK_INVERTS.get(col, False), key=f"invert_{col}")
    with st.sidebar.expander("Reward Metrics Inversion"):
        for col in reward_cols:
            if col in st.session_state.df_clean.columns:
                st.checkbox(f"Invert {col}", value=DEFAULT_REWARD_INVERTS.get(col, False), key=f"invert_{col}")

    # Metric type config
    metric_type_editor()

    # Weights
    st.sidebar.header("⚖️ Weight Configuration")
    risk_cols, reward_cols = get_current_metric_types()
    with st.sidebar.expander("Risk Weights"):
        risk_weights = {}
        for col in risk_cols:
            if col in st.session_state.df_clean.columns:
                k = f'risk_{col}'
                if k not in st.session_state: st.session_state[k] = 1.0
                st.session_state[k] = st.slider(f"{col}", 0.0, 2.0, st.session_state[k], 0.1, key=f"risk_slider_{col}")
                risk_weights[col] = st.session_state[k]
    with st.sidebar.expander("Reward Weights"):
        reward_weights = {}
        for col in reward_cols:
            if col in st.session_state.df_clean.columns:
                k = f'reward_{col}'
                if k not in st.session_state: st.session_state[k] = 1.0
                st.session_state[k] = st.slider(f"{col}", 0.0, 2.0, st.session_state[k], 0.1, key=f"reward_slider_{col}")
                reward_weights[col] = st.session_state[k]

    # Scaling with inversion
    for col in set(risk_cols + reward_cols):
        if col in st.session_state.df_clean.columns:
            invert = st.session_state.get(f"invert_{col}", DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols else DEFAULT_REWARD_INVERTS.get(col, False))
            if "Year" in st.session_state.df_clean.columns:
                st.session_state.df_clean[f"Scaled_{col}"] = st.session_state.df_clean.groupby("Year")[col].transform(lambda s: safe_min_max_scale(s, invert))
            else:
                st.session_state.df_clean[f"Scaled_{col}"] = safe_min_max_scale(st.session_state.df_clean[col], invert)

    # Scores
    current_df = calculate_scores(st.session_state.df_clean.copy(), risk_weights, reward_weights)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "👥 Manager Selection", "✏️ Edit Metrics"])
    with tab1:
        st.header("Risk-Reward Matrix")
        filtered_df = current_df.copy()
        if "Manager Name" in filtered_df.columns:
            hidden = st.session_state.get("hidden_managers", set())
            filtered_df = filtered_df[~filtered_df["Manager Name"].isin(hidden)]

        highlighted = st.multiselect("Highlight managers", filtered_df["Manager Name"].unique() if "Manager Name" in filtered_df.columns else [], max_selections=4)
        if "Manager Name" in filtered_df.columns:
            filtered_df["Highlight"] = np.where(filtered_df["Manager Name"].isin(highlighted), filtered_df["Manager Name"], "Other")
        else:
            filtered_df["Highlight"] = "Other"

        palette = px.colors.qualitative.Bold
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(highlighted)}
        color_map["Other"] = "#CCCCCC"

        args = dict(
            x="Risk_Score", y="Reward_Score", size="Bubble_Size", color="Highlight",
            hover_name="Manager Name" if "Manager Name" in filtered_df.columns else None,
            hover_data={"Risk_Score":":.3f","Reward_Score":":.3f","Average_Score":":.3f","Deal Count":True if "Deal Count" in filtered_df.columns else False,"AUM":":,.0f" if "AUM" in filtered_df.columns else False},
            size_max=40, color_discrete_map=color_map
        )
        if "Year" in filtered_df.columns: args["animation_frame"] = "Year"

        fig = px.scatter(filtered_df, **args)
        fig.update_layout(xaxis_range=[-0.1,1.1], yaxis_range=[-0.1,1.1], height=600,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="#333"), margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Metrics Table", expanded=False):
            if "Year" in filtered_df.columns and "Manager Name" in filtered_df.columns:
                table_df = filtered_df.set_index(["Year","Manager Name"])
            elif "Manager Name" in filtered_df.columns:
                table_df = filtered_df.set_index("Manager Name")
            else:
                table_df = filtered_df
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
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)
                risk_cols, reward_cols = get_current_metric_types()

                export_cols = ["Manager Name","Year","Risk_Score","Reward_Score","Average_Score","Bubble_Size"]
                export_cols += [c for c in risk_cols if c in current_df.columns]
                export_cols += [c for c in reward_cols if c in current_df.columns]
                export_cols += [c for c in ["Deal Count","AUM", "Price < 80%", "Price < 70%"] if c in current_df.columns]

                export_df = current_df[[c for c in export_cols if c in current_df.columns]].copy()

                csv_path = f"output/manager_scores_{ts}.csv"
                xlsx_path = f"output/manager_scores_{ts}.xlsx"
                export_df.to_csv(csv_path, index=False, sep=";", decimal=",")

                with pd.ExcelWriter(xlsx_path) as writer:
                    export_df.to_excel(writer, sheet_name="Scores", index=False)
                st.sidebar.success("Report generated successfully!")
                c1, c2 = st.sidebar.columns(2)
                with c1:
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 CSV", f, file_name=os.path.basename(csv_path), mime="text/csv")
                with c2:
                    with open(xlsx_path, "rb") as f:
                        st.download_button("📊 Excel", f, file_name=os.path.basename(xlsx_path),
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.sidebar.error(f"Export error: {str(e)}")

if __name__ == "__main__":
    main()
