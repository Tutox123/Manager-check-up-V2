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
        /* Police et couleurs */
        html, body, .stApp {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #333333;
        }
        
        /* Titres */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.3em;
        }
        
        /* Conteneurs */
        .metric-container {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 1.5em;
            margin-bottom: 1.5em;
        }
        
        /* Boutons */
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
        
        /* Sliders */
        .stSlider {
            color: #2c3e50;
        }
        
        /* Onglets */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid #e0e0e0;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5em 1em;
            font-weight: 500;
        }
        
        /* Tableaux */
        .dataframe {
            border: 1px solid #e0e0e0;
        }
        
        /* Metric move buttons */
        .metric-move-btn {
            font-size: 0.8em;
            padding: 0.2em 0.5em;
            margin-left: 0.5em;
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

# Initial metric definitions
INITIAL_RISK_COLS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %", "WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)"
]

INITIAL_REWARD_COLS = ["WAS", "Annualized Eq Rt (%)"]

PERCENTAGE_METRICS = [
    "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Annualized Default Rate (%)", "Junior OC cushion", "IDT Cushion", "Caa %",
    "CCC % (S&P)", "Bond %", "Second Lien %", "Cov-Lite %",
    "Avg Col Quality Test/ Trigger %", "% of collateral rated B3",
    "Price < 80%", "Price < 70%", "Annualized Eq Rt (%)", "MV NAV (Equity)",
    "Equity St. Deviation"
]

BASE_METRICS = INITIAL_RISK_COLS + INITIAL_REWARD_COLS + ["Deal Count", "AUM"]

# Initialize inversion settings (now dynamic)
def get_inversion_settings(risk_cols, reward_cols):
    """Generate inversion settings based on current categories"""
    inversions = {}
    # Default risk inversions
    risk_inverts = {
        "WARF": False, "Caa/CCC Calculated %": False, "Defaulted %": False,
        "Largest industry concentration %": False, "Diversity": True,
        "Annualized Default Rate (%)": False, "Equity St. Deviation": False,
        "Junior OC cushion": True, "IDT Cushion": True, "Caa %": False,
        "CCC % (S&P)": False, "Bond %": False, "Cov-Lite %": False,
        "WA loans Price": True, "Avg Col Quality Test/ Trigger %": True,
        "% of collateral rated B3": False, "MV NAV (Equity)": True, "WAS/WARF": True
    }
    
    # Default reward inversions (usually False)
    reward_inverts = {col: False for col in reward_cols}
    
    # Apply based on current categories
    for col in risk_cols:
        inversions[col] = risk_inverts.get(col, False)
    for col in reward_cols:
        inversions[col] = reward_inverts.get(col, False)
    
    return inversions

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
            
            # Clean data
            df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
            for col in df.columns:
                if col != "Manager Name":
                    df[col] = df[col].apply(clean_numeric)
            
            st.session_state.df_clean = df
            st.session_state.file_uploaded = True
            st.sidebar.success("File loaded successfully!")
            return df
        except Exception as e:
            st.sidebar.error(f"Loading error: {str(e)}")
            return None
    return None

def calculate_scores(df, risk_weights, reward_weights, risk_cols, reward_cols):
    # Get current inversion settings
    inversion_settings = get_inversion_settings(risk_cols, reward_cols)
    
    # First ensure all metrics are scaled
    for col in risk_cols + reward_cols:
        if col in df.columns and f"Scaled_{col}" not in df.columns:
            df[f"Scaled_{col}"] = min_max_scale(df[col], inversion_settings.get(col, False))
    
    # Risk Score
    risk_score_numerator = 0
    risk_score_denominator = 0
    for col in risk_cols:
        scaled_col = f"Scaled_{col}"
        if scaled_col in df.columns:
            weight = risk_weights.get(col, 0)
            risk_score_numerator += df[scaled_col] * weight
            risk_score_denominator += weight
    
    df["Risk_Score"] = (risk_score_numerator / risk_score_denominator) if risk_score_denominator > 0 else 0.5
    
    # Reward Score
    reward_score_numerator = 0
    reward_score_denominator = 0
    for col in reward_cols:
        scaled_col = f"Scaled_{col}"
        if scaled_col in df.columns:
            weight = reward_weights.get(col, 0)
            reward_score_numerator += df[scaled_col] * weight
            reward_score_denominator += weight
    
    df["Reward_Score"] = (reward_score_numerator / reward_score_denominator) if reward_score_denominator > 0 else 0.5
    
    df["Average_Score"] = ((1-df["Risk_Score"]) + df["Reward_Score"]) / 2
    
    if "Deal Count" in df.columns and "AUM" in df.columns:
        df["Bubble_Size"] = calculate_bubble_size(df["Deal Count"], df["AUM"])
    else:
        df["Bubble_Size"] = 30
    
    return df

def metric_editor(df):
    st.header("✏️ Manager Metrics Editor")
    
    if 'df_clean' not in st.session_state:
        st.warning("Please load a file first")
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
            inversion_settings = get_inversion_settings(
                st.session_state.risk_cols, 
                st.session_state.reward_cols
            )
            
            for col in modifications:
                if col in inversion_settings:
                    st.session_state.editable_df[f"Scaled_{col}"] = min_max_scale(
                        st.session_state.editable_df[col], 
                        inversion_settings[col]
                    )
            
            # Recalculate scores
            risk_weights = {col: st.session_state.get(f'risk_{col}', 1.0) 
                          for col in st.session_state.risk_cols}
            reward_weights = {col: st.session_state.get(f'reward_{col}', 1.0) 
                            for col in st.session_state.reward_cols}
            
            updated_df = calculate_scores(
                st.session_state.editable_df,
                risk_weights,
                reward_weights,
                st.session_state.risk_cols,
                st.session_state.reward_cols
            )
            
            # Update main data
            st.session_state.df_clean = updated_df.copy()
            st.session_state.editable_df = updated_df.copy()
            st.session_state.last_update = time.time()
            
            st.success("Changes saved successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

def metric_configuration():
    st.header("⚙️ Metric Configuration")
    
    if 'df_clean' not in st.session_state:
        st.warning("Please load data first")
        return
    
    # Current categories
    risk_cols = st.session_state.get('risk_cols', INITIAL_RISK_COLS)
    reward_cols = st.session_state.get('reward_cols', INITIAL_REWARD_COLS)
    
    # Get available metrics from data
    available_metrics = [col for col in BASE_METRICS if col in st.session_state.df_clean.columns]
    
    # Create two columns for the metric lists
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")
        risk_to_move = []
        for col in risk_cols:
            if col in available_metrics:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"• {col}")
                with cols[1]:
                    if st.button("➡️", key=f"to_reward_{col}"):
                        risk_to_move.append(col)
    
    with col2:
        st.subheader("Reward Metrics")
        reward_to_move = []
        for col in reward_cols:
            if col in available_metrics:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"• {col}")
                with cols[1]:
                    if st.button("⬅️", key=f"to_risk_{col}"):
                        reward_to_move.append(col)
    
    # Update categories if any moves were requested
    if risk_to_move or reward_to_move:
        # Move metrics from risk to reward
        for col in risk_to_move:
            if col in st.session_state.risk_cols:
                st.session_state.risk_cols.remove(col)
                st.session_state.reward_cols.append(col)
                # Preserve weight
                if f'reward_{col}' not in st.session_state:
                    st.session_state[f'reward_{col}'] = st.session_state.get(f'risk_{col}', 1.0)
        
        # Move metrics from reward to risk
        for col in reward_to_move:
            if col in st.session_state.reward_cols:
                st.session_state.reward_cols.remove(col)
                st.session_state.risk_cols.append(col)
                # Preserve weight
                if f'risk_{col}' not in st.session_state:
                    st.session_state[f'risk_{col}'] = st.session_state.get(f'reward_{col}', 1.0)
        
        st.success("Categories updated!")
        st.rerun()
    
    # Display current inversion settings
    st.subheader("Inversion Settings")
    st.write("""
        Inverted metrics are scaled in reverse (higher values become lower scores).
        Risk metrics are inverted when lower values are better (e.g., Diversity).
    """)
    
    inversion_settings = get_inversion_settings(risk_cols, reward_cols)
    inv_df = pd.DataFrame({
        'Metric': list(inversion_settings.keys()),
        'Is Inverted': list(inversion_settings.values()),
        'Category': ['Risk' if m in risk_cols else 'Reward' for m in inversion_settings.keys()]
    })
    
    st.dataframe(inv_df, hide_index=True)

def manager_selection_tab():
    st.header("👥 Manager Selection")
    
    if 'df_clean' not in st.session_state:
        st.warning("Please load data first")
        return
    
    all_managers = st.session_state.df_clean["Manager Name"].unique().tolist()
    selected_managers = st.multiselect(
        "Select managers to EXCLUDE from visualization:",
        options=all_managers,
        default=st.session_state.get('excluded_managers', []),
        key="manager_exclusion"
    )
    
    st.session_state.excluded_managers = selected_managers
    
    st.info(f"Currently excluding {len(selected_managers)} managers: {', '.join(selected_managers) if selected_managers else 'None'}")
    
    if st.button("Update Selection"):
        st.success("Selection updated!")
        st.rerun()

def main():
    # Initialisation
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.excluded_managers = []
    
    # Initialize metric categories if not already set
    if 'risk_cols' not in st.session_state:
        st.session_state.risk_cols = INITIAL_RISK_COLS.copy()
    if 'reward_cols' not in st.session_state:
        st.session_state.reward_cols = INITIAL_REWARD_COLS.copy()
    
    # Load data
    if not st.session_state.file_uploaded:
        df = load_data()
        if df is None:
            return
    
    if 'df_clean' not in st.session_state:
        st.warning("Please load a CSV file")
        return
    
    # Reset button
    if st.session_state.file_uploaded and st.sidebar.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()
    
    # Weight configuration in sidebar
    st.sidebar.header("⚖️ Weight Configuration")
    
    # Risk weights
    st.sidebar.subheader("Risk Weights")
    risk_weights = {}
    for col in st.session_state.risk_cols:
        if col in st.session_state.df_clean.columns:
            weight_key = f'risk_{col}'
            if weight_key not in st.session_state:
                st.session_state[weight_key] = 1.0
            
            st.session_state[weight_key] = st.sidebar.slider(
                f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                key=f"risk_slider_{col}"
            )
            risk_weights[col] = st.session_state[weight_key]
    
    # Reward weights
    st.sidebar.subheader("Reward Weights")
    reward_weights = {}
    for col in st.session_state.reward_cols:
        if col in st.session_state.df_clean.columns:
            weight_key = f'reward_{col}'
            if weight_key not in st.session_state:
                st.session_state[weight_key] = 1.0
            
            st.session_state[weight_key] = st.sidebar.slider(
                f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                key=f"reward_slider_{col}"
            )
            reward_weights[col] = st.session_state[weight_key]
    
    # Calculate scores with current categories
    current_df = calculate_scores(
        st.session_state.df_clean.copy(),
        risk_weights,
        reward_weights,
        st.session_state.risk_cols,
        st.session_state.reward_cols
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "✏️ Edit Metrics", "⚙️ Metric Config", "👥 Manager Select"])
    
    with tab1:
        st.header("Risk-Reward Matrix")
        
        # Filter out excluded managers
        filtered_df = current_df[~current_df["Manager Name"].isin(st.session_state.excluded_managers)]
        
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
        
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='#2c3e50'),
                opacity=0.8
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Detailed Metrics Table", expanded=False):
            st.dataframe(
                filtered_df.set_index("Manager Name"),
                use_container_width=True
            )

    with tab2:
        metric_editor(current_df)
    
    with tab3:
        metric_configuration()
    
    with tab4:
        manager_selection_tab()
    
    # Export options
    st.sidebar.header("📤 Export Options")
    if st.sidebar.button("💾 Generate Report"):
        with st.spinner("Generating report..."):
            try:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)
                
                # Prepare data
                export_cols = [
                    "Manager Name", "Risk_Score", "Reward_Score", 
                    "Average_Score", "Bubble_Size"
                ] + [col for col in BASE_METRICS if col in current_df.columns]
                
                export_df = current_df[[c for c in export_cols if c in current_df.columns]]
                
                # CSV
                csv_filename = f"manager_scores_{timestamp}.csv"
                csv_path = f"output/{csv_filename}"
                export_df.to_csv(csv_path, index=False, sep=";", decimal=",")
                
                # Excel
                excel_filename = f"manager_scores_{timestamp}.xlsx"
                excel_path = f"output/{excel_filename}"
                
                with pd.ExcelWriter(excel_path) as writer:
                    export_df.to_excel(writer, sheet_name="Scores", index=False)
                    
                    # Add weights sheet
                    weights_data = []
                    for col in st.session_state.risk_cols:
                        if col in risk_weights:
                            weights_data.append({
                                "Metric": col,
                                "Weight": risk_weights[col],
                                "Type": "Risk",
                                "Inverted": get_inversion_settings(
                                    st.session_state.risk_cols,
                                    st.session_state.reward_cols
                                ).get(col, False)
                            })
                    
                    for col in st.session_state.reward_cols:
                        if col in reward_weights:
                            weights_data.append({
                                "Metric": col,
                                "Weight": reward_weights[col],
                                "Type": "Reward",
                                "Inverted": get_inversion_settings(
                                    st.session_state.risk_cols,
                                    st.session_state.reward_cols
                                ).get(col, False)
                            })
                    
                    weights_df = pd.DataFrame(weights_data)
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
