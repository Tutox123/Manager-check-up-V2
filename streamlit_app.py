diff --git a/streamlit_app.py b/streamlit_app.py
index 2fbd912b01c6a6faf301f4faf5356bf623357cdd..c0e4bf767087410fb89198e003968522acc43eca 100644
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ -1,33 +1,34 @@
-import streamlit as st
-import pandas as pd
-import numpy as np
-import plotly.express as px
-import os
-import time
-from locale import setlocale, LC_NUMERIC, atof
-import locale
+import streamlit as st
+import pandas as pd
+import numpy as np
+import plotly.express as px
+import os
+import time
+from locale import setlocale, LC_NUMERIC, atof
+import locale
+import re
 
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
diff --git a/streamlit_app.py b/streamlit_app.py
index 2fbd912b01c6a6faf301f4faf5356bf623357cdd..c0e4bf767087410fb89198e003968522acc43eca 100644
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ -122,131 +123,149 @@ PERCENTAGE_METRICS = [
 
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
 
-def load_data():
-    st.sidebar.header("📤 Data Upload")
-    uploaded_file = st.sidebar.file_uploader(
-        "Upload your CSV file", 
-        type=["csv"],
-        help="Required format: CSV with ';' separator and ',' decimals"
-    )
-    
-    if uploaded_file is not None:
-        try:
-            df = pd.read_csv(uploaded_file, sep=";", decimal=",", thousands=' ')
-            st.session_state.df_raw = df.copy()
-            
-            # Data cleaning
-            df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
-            for col in df.columns:
-                if col != "Manager Name":
-                    df[col] = df[col].apply(clean_numeric)
-            
-            st.session_state.df_clean = df
-            st.session_state.file_uploaded = True
-            st.session_state.hidden_managers = set()
-            st.session_state.excluded_managers = set()
-            st.sidebar.success("File loaded successfully!")
-            return df
-        except Exception as e:
-            st.sidebar.error(f"Loading error: {str(e)}")
-            return None
-    return None
+def load_data():
+    st.sidebar.header("📤 Data Upload")
+    uploaded_files = st.sidebar.file_uploader(
+        "Upload your CSV files",
+        type=["csv"],
+        accept_multiple_files=True,
+        help="Required format: CSV with ';' separator and ',' decimals"
+    )
+
+    if uploaded_files:
+        all_frames = []
+        for uploaded_file in uploaded_files:
+            try:
+                df = pd.read_csv(uploaded_file, sep=";", decimal=",", thousands=' ')
+
+                # Extract year from filename
+                year_match = re.search(r"(19|20)\d{2}", uploaded_file.name)
+                year = int(year_match.group()) if year_match else 0
+                df["Year"] = year
+
+                # Data cleaning
+                df["Manager Name"] = df["Manager Name"].apply(clean_numeric, is_manager_name=True)
+                for col in df.columns:
+                    if col not in ["Manager Name", "Year"]:
+                        df[col] = df[col].apply(clean_numeric)
+
+                all_frames.append(df)
+            except Exception as e:
+                st.sidebar.error(f"Loading error for {uploaded_file.name}: {str(e)}")
+                return None
+
+        df_all = pd.concat(all_frames, ignore_index=True)
+        st.session_state.df_raw = df_all.copy()
+        st.session_state.df_clean = df_all
+        st.session_state.file_uploaded = True
+        st.session_state.hidden_managers = set()
+        st.session_state.excluded_managers = set()
+        st.sidebar.success("Files loaded successfully!")
+        return df_all
+    return None
 
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
     
-    working_df["Average_Score"] = ((1-working_df["Risk_Score"]) + working_df["Reward_Score"]) / 2
-    
-    if "Deal Count" in working_df.columns and "AUM" in working_df.columns:
-        working_df["Bubble_Size"] = calculate_bubble_size(working_df["Deal Count"], working_df["AUM"])
-    else:
-        working_df["Bubble_Size"] = 30
-    
-    # Merge back with original dataframe
-    df = df.merge(working_df[["Manager Name", "Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]], 
-                 on="Manager Name", how="left", suffixes=('', '_y'))
-    
-    return df
+    working_df["Average_Score"] = ((1-working_df["Risk_Score"]) + working_df["Reward_Score"]) / 2
+
+    if "Deal Count" in working_df.columns and "AUM" in working_df.columns:
+        working_df["Bubble_Size"] = calculate_bubble_size(working_df["Deal Count"], working_df["AUM"])
+    else:
+        working_df["Bubble_Size"] = 30
+
+    # Merge back with original dataframe
+    merge_cols = ["Manager Name"]
+    if "Year" in df.columns:
+        merge_cols.append("Year")
+    df = df.merge(
+        working_df[merge_cols + ["Risk_Score", "Reward_Score", "Average_Score", "Bubble_Size"]],
+        on=merge_cols,
+        how="left",
+        suffixes=('', '_y')
+    )
+
+    return df
 
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
diff --git a/streamlit_app.py b/streamlit_app.py
index 2fbd912b01c6a6faf301f4faf5356bf623357cdd..c0e4bf767087410fb89198e003968522acc43eca 100644
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ -475,134 +494,164 @@ def main():
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
-    for col in risk_cols + reward_cols:
-        if col in st.session_state.df_clean.columns:
-            invert = st.session_state.get(f"invert_{col}", 
-                                       DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols 
-                                       else DEFAULT_REWARD_INVERTS.get(col, False))
-            st.session_state.df_clean[f"Scaled_{col}"] = min_max_scale(
-                st.session_state.df_clean[col], 
-                invert
-            )
+    for col in risk_cols + reward_cols:
+        if col in st.session_state.df_clean.columns:
+            invert = st.session_state.get(
+                f"invert_{col}",
+                DEFAULT_RISK_INVERTS.get(col, False) if col in risk_cols
+                else DEFAULT_REWARD_INVERTS.get(col, False)
+            )
+            if "Year" in st.session_state.df_clean.columns:
+                st.session_state.df_clean[f"Scaled_{col}"] = st.session_state.df_clean.groupby("Year")[col].transform(
+                    lambda s: min_max_scale(s, invert)
+                )
+            else:
+                st.session_state.df_clean[f"Scaled_{col}"] = min_max_scale(
+                    st.session_state.df_clean[col],
+                    invert
+                )
     
     # Score calculation
     current_df = calculate_scores(
         st.session_state.df_clean.copy(),
         risk_weights,
         reward_weights
     )
     
     # Tabs
     tab1, tab2, tab3 = st.tabs(["📊 Visualization", "👥 Manager Selection", "✏️ Edit Metrics"])
     
-    with tab1:
-        st.header("Risk-Reward Matrix")
-        
-        # Filter managers based on visibility settings
-        filtered_df = current_df[~current_df["Manager Name"].isin(st.session_state.get("hidden_managers", set()))]
-        
-        fig = px.scatter(
-            filtered_df,
-            x="Risk_Score",
-            y="Reward_Score",
-            size="Bubble_Size",
-            color="Average_Score",
-            color_continuous_scale="blues",
-            hover_name="Manager Name",
-            hover_data={
-                "Risk_Score": ":.3f",
-                "Reward_Score": ":.3f",
-                "Average_Score": ":.3f",
-                "Deal Count": True,
-                "AUM": ":,.0f"
-            },
-            size_max=40
-        )
+    with tab1:
+        st.header("Risk-Reward Matrix")
+
+        # Filter managers based on visibility settings
+        filtered_df = current_df[~current_df["Manager Name"].isin(st.session_state.get("hidden_managers", set()))].copy()
+
+        highlighted = st.multiselect(
+            "Highlight managers",
+            filtered_df["Manager Name"].unique(),
+            max_selections=4
+        )
+
+        filtered_df["Highlight"] = np.where(
+            filtered_df["Manager Name"].isin(highlighted),
+            filtered_df["Manager Name"],
+            "Other"
+        )
+
+        color_map = {m: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
+                     for i, m in enumerate(highlighted)}
+        color_map["Other"] = "#CCCCCC"
+
+        scatter_args = dict(
+            x="Risk_Score",
+            y="Reward_Score",
+            size="Bubble_Size",
+            color="Highlight",
+            hover_name="Manager Name",
+            hover_data={
+                "Risk_Score": ":.3f",
+                "Reward_Score": ":.3f",
+                "Average_Score": ":.3f",
+                "Deal Count": True,
+                "AUM": ":,.0f"
+            },
+            size_max=40,
+            color_discrete_map=color_map
+        )
+        if "Year" in filtered_df.columns:
+            scatter_args["animation_frame"] = "Year"
+
+        fig = px.scatter(filtered_df, **scatter_args)
         
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
         
-        with st.expander("Detailed Metrics Table", expanded=False):
-            st.dataframe(
-                filtered_df.set_index("Manager Name"),
-                use_container_width=True
-            )
+        with st.expander("Detailed Metrics Table", expanded=False):
+            if "Year" in filtered_df.columns:
+                table_df = filtered_df.set_index(["Year", "Manager Name"])
+            else:
+                table_df = filtered_df.set_index("Manager Name")
+            st.dataframe(
+                table_df,
+                use_container_width=True
+            )
     
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
-                export_cols = [
-                    "Manager Name", "Risk_Score", "Reward_Score", 
-                    "Average_Score", "Bubble_Size"
-                ] + risk_cols + reward_cols + ["Deal Count", "AUM"]
+                export_cols = [
+                    "Manager Name", "Year", "Risk_Score", "Reward_Score",
+                    "Average_Score", "Bubble_Size"
+                ] + risk_cols + reward_cols + ["Deal Count", "AUM"]
                 
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
