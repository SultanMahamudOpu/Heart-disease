import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap 

# 1. Page Config & Custom CSS
st.set_page_config(page_title="CVD DSS System", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    
    /* Input label size increase */
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p {
        font-size: 25px !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 Cardiovascular Disease Decision Support System")
st.markdown("---")

# Helper function to render SHAP plot in Streamlit
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 2. Load Assets safely
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_cvd_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}. Please run the notebook code first to save the model and assets.")
    st.stop()

# 3. Main Panel Layout
col_inputs, col_outputs = st.columns([2, 3])

# --- Input Panel ---
with col_inputs:
    st.header("📋 Patient Input Panel")
    
    with st.expander("🔴 High-Impact Risk Factors", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20.0, 100.0, 50.0)
            family = st.selectbox("Family History of CVD", ["N", "Y"])
            smoking = st.selectbox("Smoking Status", ["N", "Y"])
            diabetes = st.selectbox("Diabetes Status", ["N", "Y"])
            sex = st.selectbox("Sex", ["F", "M"])
        with col2:
            sys_bp = st.number_input("Systolic BP (mmHg)", 80.0, 250.0, 130.0)
            activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            height = st.number_input("Height (cm)", 100.0, 220.0, 165.0)
            bmi = weight / ((height / 100) ** 2) if height > 0 else 0
            st.info(f"Calculated BMI: **{bmi:.2f}**")

    with st.expander("🟡 Secondary Clinical Metrics", expanded=False):
        col3, col4 = st.columns(2)
        with col3:
            hdl = st.number_input("HDL (mg/dL)", 20.0, 100.0, 45.0)
            dia_bp = st.number_input("Diastolic BP (mmHg)", 50.0, 130.0, 80.0)
            bp_cat = st.selectbox("Blood Pressure Category", ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"])
        with col4:
            chol = st.number_input("Total Cholesterol (mg/dL)", 100.0, 400.0, 200.0)
            fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 50.0, 300.0, 95.0)
            ldl = st.number_input("Estimated LDL (mg/dL)", 50.0, 250.0, 120.0)
            abdominal = st.number_input("Abdominal Circumference (cm)", 50.0, 150.0, 90.0)
            wh_ratio = abdominal / height if height > 0 else 0
            st.info(f"Waist-Height Ratio: **{wh_ratio:.3f}**")

    predict_btn = st.button("Predict CVD Risk", use_container_width=True, type="primary")

# --- Output Panel ---
with col_outputs:
    st.header("⚙️ Real-Time Assessment")
    
    if predict_btn:
        input_dict = {
            'Age': age, 'Weight (kg)': weight, 'Height (cm)': height, 
            'BMI': bmi, 'Abdominal Circumference (cm)': abdominal, 'Total Cholesterol (mg/dL)': chol,
            'HDL (mg/dL)': hdl, 'Fasting Blood Sugar (mg/dL)': fbs, 'Waist-to-Height Ratio': wh_ratio,
            'Systolic BP': sys_bp, 'Diastolic BP': dia_bp, 'Estimated LDL (mg/dL)': ldl,
            'Sex': sex, 'Smoking Status': smoking, 'Diabetes Status': diabetes,
            'Physical Activity Level': activity, 'Family History of CVD': family,
            'Blood Pressure Category': bp_cat
        }
        temp_df = pd.DataFrame([input_dict])

        mapping_dict = {
            'Sex': {'F': 0, 'M': 1}, 'Smoking Status': {'N': 0, 'Y': 1}, 'Diabetes Status': {'N': 0, 'Y': 1},
            'Family History of CVD': {'N': 0, 'Y': 1}, 'Physical Activity Level': {'High': 0, 'Low': 1, 'Moderate': 2},
            'Blood Pressure Category': {'Elevated': 0, 'Hypertension Stage 1': 1, 'Hypertension Stage 2': 2, 'Normal': 3}
        }
        for col, map_val in mapping_dict.items():
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].map(map_val)
        
        try:
            user_data = temp_df[expected_columns]
        except Exception as e:
            st.error(f"Missing columns in input: {e}")
            st.stop()
            
        user_data_scaled = scaler.transform(user_data)

        prediction = model.predict(user_data_scaled)[0]
        risk_prob = model.predict_proba(user_data_scaled)[0]
        
        # ---------------------------------------------------------
        # Pre-calculating SHAP values 
        # ---------------------------------------------------------
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_data_scaled)
            
            # Fixing data types and dimensions
            if isinstance(shap_values, list):
                target_shap_raw = shap_values[prediction][0]
                base_value = explainer.expected_value[prediction]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                target_shap_raw = shap_values[0, :, prediction]
                base_value = explainer.expected_value[prediction]
            else:
                target_shap_raw = shap_values[0]
                base_value = explainer.expected_value
                
            # Forcing data to 1D array
            target_shap = np.array(target_shap_raw).flatten()
            
        except Exception as e:
            target_shap = None
            st.error(f"Error in SHAP processing: {e}")

        # [A] Final Risk Prediction
        st.subheader("1. Final CVD Risk Prediction")
        
        if prediction == 2:
            st.markdown("""
                <div style="
                    background-color: #bd1c1c; 
                    color: white; 
                    padding: 20px; 
                    text-align: center; 
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                    margin-bottom: 15px;">
                    <h1 style="color: white; margin: 0; font-size: 45px; font-weight: 800; letter-spacing: 2px;">
                        🚨 HIGH RISK DETECTED 🚨
                    </h1>
                </div>
            """, unsafe_allow_html=True)
            st.write("Immediate medical consultation advised.")
            
        elif prediction == 1:
            st.markdown("""
                <div style="
                    background-color: #d97706; 
                    color: white; 
                    padding: 20px; 
                    text-align: center; 
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                    margin-bottom: 15px;">
                    <h1 style="color: white; margin: 0; font-size: 40px; font-weight: 800;">
                        ⚠️ INTERMEDIARY RISK ⚠️
                    </h1>
                </div>
            """, unsafe_allow_html=True)
            st.write("Aggressive lifestyle modification recommended.")
            
        else:
            st.markdown("""
                <div style="
                    background-color: #16a34a; 
                    color: white; 
                    padding: 20px; 
                    text-align: center; 
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                    margin-bottom: 15px;">
                    <h1 style="color: white; margin: 0; font-size: 40px; font-weight: 800;">
                        ✅ LOW RISK ✅
                    </h1>
                </div>
            """, unsafe_allow_html=True)
            st.write("Maintain healthy lifestyle.")
            
        st.markdown("---")
        
        # [B] Probability Breakdown
        st.subheader("2. Probability Breakdown")
        prob_labels = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk'] 
        fig_prob = px.bar(x=prob_labels, y=risk_prob * 100, 
                          text=[f"{val:.1f}%" for val in (risk_prob * 100)],
                          labels={'x': 'Risk Category', 'y': 'Probability (%)'},
                          color=prob_labels, color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
                          
        fig_prob.update_traces(
            textposition='outside',
            textfont_size=40,        
            textfont_color='black'   
        )
        
        # অক্ষের সংশোধিত কনফিগারেশন (Error Fixed)
        fig_prob.update_layout(
            yaxis_range=[0, 115],
            xaxis=dict(
                tickfont=dict(size=22, family="sans-serif", weight="bold"),  # font_family বদলে family এবং বোল্ড করা হয়েছে
                title=dict(font=dict(size=24))                     
            ),
            yaxis=dict(
                tickfont=dict(size=18),                            
                title=dict(font=dict(size=24))                     
            ),
            showlegend=False                                       
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("---")

        # [C] Dynamic Radar Chart
        st.subheader("3. Top 5 Risk Drivers (Dynamic Radar)")
        if target_shap is not None:
            try:
                impact_df = pd.DataFrame({
                    'Feature': expected_columns, 
                    'Impact': np.abs(target_shap) 
                })
                top_5_impact = impact_df.sort_values(by='Impact', ascending=False).head(5)
                
                top_features = top_5_impact['Feature'].tolist()
                impact_scores = top_5_impact['Impact'].tolist()
                
                radar_labels = []
                for feat in top_features:
                    val = input_dict.get(feat, '')
                    if isinstance(val, float):
                        val = round(val, 2)
                    radar_labels.append(f"{feat}<br><b>({val})</b>")
                
                fig_radar = go.Figure(go.Scatterpolar(
                    r=impact_scores, 
                    theta=radar_labels, 
                    fill='toself', 
                    line_color='#0d9488',
                    name='Impact Score'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating radar chart: {e}")

        st.markdown("---")

        # [D] SHAP Force Plot
        st.subheader("4. Feature Contribution (SHAP Force Plot)")
        st.caption("Red features increase the risk (pushing right), while blue features decrease the risk (pushing left).")
        if target_shap is not None:
            try:
                force_plot = shap.force_plot(
                    base_value,          
                    target_shap,         
                    user_data.iloc[0, :], 
                    feature_names=expected_columns,
                    matplotlib=False
                )
                st_shap(force_plot, height=150)
            except Exception as e:
                st.error(f"Error loading SHAP plot: {e}")

    else:
        st.info("👈 Enter patient data in the left panel and click 'Predict CVD Risk'.")