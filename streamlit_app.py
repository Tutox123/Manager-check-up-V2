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

# Style CSS sobre
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
    </style>
""", unsafe_allow_html=True)

# Titre sobre
st.markdown("<h1 style='text-align: center; margin-bottom: 1em;'>Manager Risk-Reward Analysis</h1>", unsafe_allow_html=True)

# Fonctions utilitaires (inchangées)
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
PERCENTAGE_METRICS = [
    "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Annualized Default Rate (%)", "Junior OC cushion", "IDT Cushion", "Caa %",
    "CCC % (S&P)", "Bond %", "Second Lien %", "Cov-Lite %",
    "Avg Col Quality Test/ Trigger %", "% of collateral rated B3",
    "Price < 80%", "Price < 70%", "Annualized Eq Rt (%)", "MV NAV (Equity)",
    "Equity St. Deviation"
]

RISK_COLS = [
    "WARF", "Caa/CCC Calculated %", "Defaulted %", "Largest industry concentration %",
    "Diversity", "Annualized Default Rate (%)", "Equity St. Deviation",
    "Junior OC cushion", "IDT Cushion", "Caa %", "CCC % (S&P)", "Bond %",
    "Cov-Lite %", "WA loans Price", "Avg Col Quality Test/ Trigger %","WAS/WARF",
    "% of collateral rated B3", "MV NAV (Equity)"
]

REWARD_COLS = ["WAS", "Annualized Eq Rt (%)"]
BASE_METRICS = RISK_COLS + REWARD_COLS + ["Deal Count", "AUM"]

RISK_INVERTS = {
    "WARF": False, "Caa/CCC Calculated %": False, "Defaulted %": False,
    "Largest industry concentration %": False, "Diversity": True,
    "Annualized Default Rate (%)": False, "Equity St. Deviation": False,
    "Junior OC cushion": True, "IDT Cushion": True, "Caa %": False,
    "CCC % (S&P)": False, "Bond %": False, "Cov-Lite %": False,
    "WA loans Price": True, "Avg Col Quality Test/ Trigger %": True,
    "% of collateral rated B3": False, "MV NAV (Equity)": True,"WAS/WARF": True
}

REWARD_INVERTS = {col: False for col in REWARD_COLS}

def load_data():
    st.sidebar.header("📤 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=["csv"],
        help="Format requis : CSV avec séparateur ';' et décimales ','"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";", decimal=",", thousands=' ')
            st.session_state.df_raw = df.copy()
            
            # Nettoyage des données
            df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
            for col in df.columns:
                if col != "Manager Name":
                    df[col] = df[col].apply(clean_numeric)
            
            st.session_state.df_clean = df
            st.session_state.file_uploaded = True
            st.sidebar.success("Fichier chargé avec succès!")
            return df
        except Exception as e:
            st.sidebar.error(f"Erreur de chargement : {str(e)}")
            return None
    return None

def calculate_scores(df, risk_weights, reward_weights):
    # Risk Score
    risk_score_numerator = 0
    risk_score_denominator = 0
    for col in RISK_COLS:
        scaled_col = f"Scaled_{col}"
        if scaled_col in df.columns:
            weight = risk_weights.get(col, 0)
            risk_score_numerator += df[scaled_col] * weight
            risk_score_denominator += weight
    
    df["Risk_Score"] = (risk_score_numerator / risk_score_denominator) if risk_score_denominator > 0 else 0.5
    
    # Reward Score
    reward_score_numerator = 0
    reward_score_denominator = 0
    for col in REWARD_COLS:
        scaled_col = f"Scaled_{col}"
        if scaled_col in df.columns:
            weight = reward_weights.get(col, 0)
            reward_score_numerator += df[scaled_col] * weight
            reward_score_denominator += weight
    
    df["Reward_Score"] = (reward_score_numerator / reward_score_denominator) if reward_score_denominator > 0 else 0.5
    
    df["Average_Score"] = (df["Risk_Score"] + df["Reward_Score"]) / 2
    
    if "Deal Count" in df.columns and "AUM" in df.columns:
        df["Bubble_Size"] = calculate_bubble_size(df["Deal Count"], df["AUM"])
    else:
        df["Bubble_Size"] = 30
    
    return df

def metric_editor(df):
    st.header("✏️ Manager Metrics Editor")
    
    if 'df_clean' not in st.session_state:
        st.warning("Veuillez d'abord charger un fichier")
        return
    
    if 'editable_df' not in st.session_state:
        st.session_state.editable_df = st.session_state.df_clean.copy()
    
    selected_manager = st.selectbox(
        "Sélectionnez un manager",
        st.session_state.editable_df["Manager Name"].unique(),
        key="manager_select"
    )
    
    manager_idx = st.session_state.editable_df[
        st.session_state.editable_df["Manager Name"] == selected_manager
    ].index[0]
    
    st.markdown(f"### Modification pour : **{selected_manager}**")
    st.markdown("<div class='metric-modif'>", unsafe_allow_html=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("💾 Enregistrer", key=f"save_{selected_manager}"):
        try:
            # Appliquer les modifications
            for col, new_value in modifications.items():
                st.session_state.editable_df.at[manager_idx, col] = new_value
            
            # Mettre à jour les colonnes scaled
            for col in set(modifications.keys()) & set(RISK_COLS + REWARD_COLS):
                invert = RISK_INVERTS.get(col, False) if col in RISK_COLS else REWARD_INVERTS.get(col, False)
                st.session_state.editable_df[f"Scaled_{col}"] = min_max_scale(
                    st.session_state.editable_df[col], 
                    invert
                )
            
            # Recalculer les scores
            risk_weights = {col: st.session_state.get(f'risk_{col}', 1.0) for col in RISK_COLS}
            reward_weights = {col: st.session_state.get(f'reward_{col}', 1.0) for col in REWARD_COLS}
            
            updated_df = calculate_scores(
                st.session_state.editable_df,
                risk_weights,
                reward_weights
            )
            
            # Mettre à jour les données principales
            st.session_state.df_clean = updated_df.copy()
            st.session_state.editable_df = updated_df.copy()
            st.session_state.last_update = time.time()
            
            st.success("Modifications enregistrées avec succès!")
            
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

def main():
    # Initialisation
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    # Chargement des données
    if not st.session_state.file_uploaded:
        df = load_data()
        if df is None:
            return
    
    if 'df_clean' not in st.session_state:
        st.warning("Veuillez charger un fichier CSV")
        return
    
    # Bouton de réinitialisation
    if st.session_state.file_uploaded and st.sidebar.button("🔄 Réinitialiser"):
        st.session_state.clear()
        st.rerun()
    
    # Scaling des données
    for col in RISK_COLS:
        if col in st.session_state.df_clean.columns:
            st.session_state.df_clean[f"Scaled_{col}"] = min_max_scale(
                st.session_state.df_clean[col], 
                RISK_INVERTS.get(col, False)
            )
    
    for col in REWARD_COLS:
        if col in st.session_state.df_clean.columns:
            st.session_state.df_clean[f"Scaled_{col}"] = min_max_scale(
                st.session_state.df_clean[col], 
                REWARD_INVERTS.get(col, False)
            )
    
    # Configuration des poids
    st.sidebar.header("⚖️ Weight Configuration")
    
    st.sidebar.subheader("Risk Weights")
    risk_weights = {}
    for col in RISK_COLS:
        if col in st.session_state.df_clean.columns:
            weight_key = f'risk_{col}'
            if weight_key not in st.session_state:
                st.session_state[weight_key] = 1.0
            
            st.session_state[weight_key] = st.sidebar.slider(
                f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                key=f"risk_slider_{col}"
            )
            risk_weights[col] = st.session_state[weight_key]
    
    st.sidebar.subheader("Reward Weights")
    reward_weights = {}
    for col in REWARD_COLS:
        if col in st.session_state.df_clean.columns:
            weight_key = f'reward_{col}'
            if weight_key not in st.session_state:
                st.session_state[weight_key] = 1.0
            
            st.session_state[weight_key] = st.sidebar.slider(
                f"{col}", 0.0, 2.0, st.session_state[weight_key], 0.1,
                key=f"reward_slider_{col}"
            )
            reward_weights[col] = st.session_state[weight_key]
    
    # Calcul des scores
    current_df = calculate_scores(
        st.session_state.df_clean.copy(),
        risk_weights,
        reward_weights
    )
    
    # Onglets
    tab1, tab2 = st.tabs(["📊 Visualization", "✏️ Edit Metrics"])
    with tab1:
        st.header("Risk-Reward Matrix")
        
        fig = px.scatter(
            current_df,
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
                current_df.set_index("Manager Name"),
                use_container_width=True
            )

    with tab2:
        metric_editor(current_df)
    
    # Export des résultats
    st.sidebar.header("📤 Export Options")
    if st.sidebar.button("💾 Generate Report"):
        with st.spinner("Génération du rapport..."):
            try:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output", exist_ok=True)
                
                # Préparation des données
                export_cols = [
                    "Manager Name", "Risk_Score", "Reward_Score", 
                    "Average_Score", "Bubble_Size"
                ] + BASE_METRICS
                
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
                    
                    weights_df = pd.DataFrame({
                        "Metric": list(risk_weights.keys()) + list(reward_weights.keys()),
                        "Weight": list(risk_weights.values()) + list(reward_weights.values()),
                        "Type": ["Risk"]*len(risk_weights) + ["Reward"]*len(reward_weights)
                    })
                    weights_df.to_excel(writer, sheet_name="Weights", index=False)
                
                # Téléchargement
                st.sidebar.success("Rapport généré avec succès!")
                
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
                st.sidebar.error(f"Erreur d'export : {str(e)}")

if __name__ == "__main__":
    main()
