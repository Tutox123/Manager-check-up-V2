import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import time
from locale import setlocale, LC_NUMERIC, atof
import locale

# Configuration initiale
try:
    setlocale(LC_NUMERIC, 'en_US.UTF-8')
except:
    setlocale(LC_NUMERIC, '')

# Configuration de la page
st.set_page_config(
    layout="wide", 
    page_title="Manager Risk Analysis",
    page_icon="📈"
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
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=["csv"],
        help="Required format: CSV with ';' separator and ',' decimals"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";", decimal=",", thousands=' ')
            st.session_state.df_raw = df.copy()
            
            # Data cleaning
            df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
            for col in df.columns:
                if col != "Manager Name":
                    df[col] = df[col].apply(clean_numeric)
            
            st.session_state.df_clean = df
            st.session_state.file_uploaded = True
            st.session_state.hidden_managers = set()
            st.session_state.excluded_managers = set()
            st.sidebar.success("File loaded successfully!")
            return df
        except Exception as e:
            st.sidebar.error(f"Loading error: {str(e)}")
            return None
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
    df = df.merge(working_df[["Manager Name", "Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]], 
                 on="Manager Name", how="left", suffixes=('', '_y'))
    
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
        key="manager_select"
    )
    
    manager_idx = st.session_state.editable_df[
        st.session_state.editable_df["Manager Name"] == selected_manager
    ].index[0]
    
    st.markdown(f"### Editing: **{selected_manager}**")
    
    cols = st.columns(3)
    current_col = 0
    modifications = {}
    
    for col in BASE_METRICS:
        if col in st.session_state.editable_df.columns:
            with cols[current_col]:
                current_val = st.session_state.editable_df.at[manager_idx, col]
                
                if col in PERCENTAGE_METRICS:
                    new_val = st.number_input(
                        f"{col} (%)",
                        value=float(current_val * 100),
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key=f"edit_{col}_{selected_manager}"
                    )
                    modifications[col] = new_val / 100
                else:
                    new_val = st.number_input(
                        col,
                        value=float(current_val),
                        step=0.1,
                        key=f"edit_{col}_{selected_manager}"
                    )
                    modifications[col] = new_val
            
            current_col = (current_col + 1) % 3
    
    if st.button("💾 Save", key=f"save_{selected_manager}"):
        try:
            # Apply modifications
            for col, new_value in modifications.items():
                st.session_state.editable_df.at[manager_idx, col] = new_value
            
            # Update scaled columns
            risk_cols, reward_cols = get_current_metric_types()
            for col in set(modifications.keys()) & set(risk_cols + reward_cols):
                invert = st.session_state.get(f"invert_{col}", 
                                           DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols 
                                           else DEFAULT_REWARD_INVERTS.get(col, False))
                st.session_state.editable_df[f"Scaled_{col}"] = min_max_scale(
                    st.session_state.editable_df[col], 
                    invert
                )
            
            # Recalculate scores
            risk_weights = {col: st.session_state.get(f'risk_{col}', 1.0) for col in risk_cols}
            reward_weights = {col: st.session_state.get(f'reward_{col}', 1.0) for col in reward_cols}
            
            updated_df = calculate_scores(
                st.session_state.editable_df,
                risk_weights,
                reward_weights
            )
            
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
    
    all_managers = st.session_state.df_clean["Manager Name"].unique()
    
    # Two columns for the two lists
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hide from Visualization")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        
        hidden_managers = st.session_state.get("hidden_managers", set())
        new_hidden = set()
        
        for manager in all_managers:
            checked = st.checkbox(
                f"Hide {manager}",
                value=manager in hidden_managers,
                key=f"hide_{manager}"
            )
            if checked:
                new_hidden.add(manager)
        
        st.session_state.hidden_managers = new_hidden
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Exclude from Calculations")
        st.markdown("<div class='manager-list'>", unsafe_allow_html=True)
        
        excluded_managers = st.session_state.get("excluded_managers", set())
        new_excluded = set()
        
        for manager in all_managers:
            checked = st.checkbox(
                f"Exclude {manager}",
                value=manager in excluded_managers,
                key=f"exclude_{manager}"
            )
            if checked:
                new_excluded.add(manager)
        
        st.session_state.excluded_managers = new_excluded
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Initialization
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.custom_risk_cols = DEFAULT_RISK_COLS.copy()
        st.session_state.custom_reward_cols = DEFAULT_REWARD_COLS.copy()
    
    # Data loading
    if not st.session_state.file_uploaded:
        df = load_data()
        if df is None:
            return
    
    if 'df_clean' not in st.session_state:
        st.warning("Please upload a CSV file")
        return
    
    # Reset button
    if st.session_state.file_uploaded and st.sidebar.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()
    
    # Metric inversion UI
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
    
    # Metric type configuration
    metric_type_editor()
    
    # Weight configuration
    st.sidebar.header("⚖️ Weight Configuration")
    
    risk_cols, reward_cols = get_current_metric_types()
    
    with st.sidebar.expander("Risk Weights"):
        risk_weights = {}
        for col in risk_cols:
            if col in st.session_state.df_clean.columns:
                weight_key = f'risk_{col}'
                if weight_key not in st.session_state:
                    st.session_state[weight_key] = 1.0
                
                st.session_state[weight_key] = st.slider(
                    f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                    key=f"risk_slider_{col}"
                )
                risk_weights[col] = st.session_state[weight_key]
    
    with st.sidebar.expander("Reward Weights"):
        reward_weights = {}
        for col in reward_cols:
            if col in st.session_state.df_clean.columns:
                weight_key = f'reward_{col}'
                if weight_key not in st.session_state:
                    st.session_state[weight_key] = 1.0
                
                st.session_state[weight_key] = st.slider(
                    f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                    key=f"reward_slider_{col}"
                )
                reward_weights[col] = st.session_state[weight_key]
    
    # Data scaling with current inversion settings
    for col in risk_cols + reward_cols:
        if col in st.session_state.df_clean.columns:
            invert = st.session_state.get(f"invert_{col}", 
                                       DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols 
                                       else DEFAULT_REWARD_INVERTS.get(col, False))
            st.session_state.df_clean[f"Scaled_{col}"] = min_max_scale(
                st.session_state.df_clean[col], 
                invert
            )
    
    # Score calculation
    current_df = calculate_scores(
        st.session_state.df_clean.copy(),
        risk_weights,
        reward_weights
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "👥 Manager Selection", "✏️ Edit Metrics"])
    
    with tab1:
        st.header("Risk-Reward Matrix")
        
        # Filter managers based on visibility settings
        filtered_df = current_df[~current_df["Manager Name"].isin(st.session_state.get("hidden_managers", set()))]
        
        fig = px.scatter(
            filtered_df,
            x="Risk_Score",
            y="Reward_Score",
            size="Bubble_Size",
            color="Average_Score",
            color_continuous_scale="blues",
            hover_name="Manager Name",
            hover_data={
                "Risk_Score": ":.3f",
                "Reward_Score": ":.3f",
                "Average_Score": ":.3f",
                "Deal Count": True,
                "AUM": ":,.0f"
            },
            size_max=40
        )
        
        fig.update_layout(
            xaxis_range=[-0.1, 1.1],
            yaxis_range=[-0.1, 1.1],
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333333"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Detailed Metrics Table", expanded=False):
            st.dataframe(
                filtered_df.set_index("Manager Name"),
                use_container_width=True
            )
    
    with tab2:
        manager_selection_interface()
    
    with tab3:
        metric_editor(current_df)
    
    # Export options
    st.sidebar.header("📤 Export Options")
    if st.sidebar.button("💾 Generate Report"):
        with st.spinner("Generating report..."):
            try:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)
                
                # Get current metric classification
                risk_cols, reward_cols = get_current_metric_types()
                
                # Prepare data
                export_cols = [
                    "Manager Name", "Risk_Score", "Reward_Score", 
                    "Average_Score", "Bubble_Size"
                ] + risk_cols + reward_cols + ["Deal Count", "AUM"]
                
                export_df = current_df[[c for c in export_cols if c in current_df.columns]]
                
                # CSV
                csv_filename = f"manager_scores_{timestamp}.csv"
                csv_path = f"output/{csv_filename}"
                export_df.to_csv(csv_path, index=False, sep=";", decimal=",")
                
                # Excel
                excel_filename = f"manager_scores_{timestamp}.xlsx"
                excel_path = f"output/{excel_filename}"
                
                with pd.ExcelWriter(excel_path) as writer:
                    # 1. Scores sheet
                    export_df.to_excel(writer, sheet_name="Scores", index=False)
                    
                    # 2. Manager Selections sheet
                    selections_df = pd.DataFrame({
                        "Manager": current_df["Manager Name"],
                        "Hidden": current_df["Manager Name"].isin(st.session_state.get("hidden_managers", set())),
                        "Excluded": current_df["Manager Name"].isin(st.session_state.get("excluded_managers", set()))
                    })
                    selections_df.to_excel(writer, sheet_name="Manager Selections", index=False)
                    
                    # 3. Metric Types sheet
                    metric_types_df = pd.DataFrame({
                        "Metric": risk_cols + reward_cols,
                        "Type": ["Risk"]*len(risk_cols) + ["Reward"]*len(reward_cols),
                        "Default Inverted": [
                            DEFAULT_RISK_INVERTS.get(col, False) if col in DEFAULT_RISK_COLS 
                            else DEFAULT_REWARD_INVERTS.get(col, False) 
                            for col in risk_cols + reward_cols
                        ]
                    })
                    metric_types_df.to_excel(writer, sheet_name="Metric Types", index=False)
                    
                    # 4. Current Inversion Settings sheet
                    inversion_df = pd.DataFrame([
                        {
                            "Metric": col,
                            "Inverted": st.session_state.get(f"invert_{col}", 
                                          DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols 
                                          else DEFAULT_REWARD_INVERTS.get(col, False)),
                            "Type": "Risk" if col in risk_cols else "Reward"
                        }
                        for col in risk_cols + reward_cols
                    ])
                    inversion_df.to_excel(writer, sheet_name="Inversion Settings", index=False)
                    
                    # 5. Weights sheet
                    weights_df = pd.DataFrame({
                        "Metric": risk_cols + reward_cols,
                        "Weight": [st.session_state.get(f'risk_{col}', 1.0) for col in risk_cols] + 
                                  [st.session_state.get(f'reward_{col}', 1.0) for col in reward_cols],
                        "Type": ["Risk"]*len(risk_cols) + ["Reward"]*len(reward_cols)
                    })
                    weights_df.to_excel(writer, sheet_name="Weights", index=False)
                
                # Download buttons
                st.sidebar.success("Report generated successfully!")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "📥 CSV",
                            f,
                            file_name=csv_filename,
                            mime="text/csv"
                        )
                
                with col2:
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            "📊 Excel",
                            f,
                            file_name=excel_filename,
                            mime="application/vnd.ms-excel"
                        )
            
            except Exception as e:
                st.sidebar.error(f"Export error: {str(e)}")

if __name__ == "__main__":
    main()
