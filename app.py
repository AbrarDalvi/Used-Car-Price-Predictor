"""
Streamlit Web Application
Project: Used Car Resale Price Prediction Using Machine Learning Algorithms
Author: College ML Project
Tech: Streamlit, Scikit-Learn, Pandas, NumPy
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Used Car Resale Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .price-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .price-amount {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .price-range {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
    .metric-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model and Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    model_path = os.path.join("models", "best_car_price_model.joblib")
    meta_path = os.path.join("models", "model_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error("Model files not found! Please run `python src/train.py` first.")
        st.stop()
        
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_model_and_metadata()
feature_info = metadata.get("feature_info", {})
metrics_summary = metadata.get("metrics_summary", [])
best_model_name = metadata.get("best_model_name", "Random Forest Regressor")

# Format currency helper (INR Lakhs & thousands)
def format_inr(number):
    if number >= 10000000:
        return f"₹ {number / 10000000:.2f} Crore"
    elif number >= 100000:
        return f"₹ {number / 100000:.2f} Lakh (₹ {number:,.0f})"
    else:
        return f"₹ {number:,.0f}"


# ---------------------------------------------------------
# Sidebar: Project & Model Benchmark Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car.png", width=70)
    st.title("Project Overview")
    st.markdown("""
    **College ML Project:**  
    *Used Car Resale Price Prediction Using Machine Learning Algorithms*
    
    **Algorithms Implemented:**
    1. Random Forest Regressor
    2. Extra Trees Regressor
    3. Gradient Boosting Regressor
    """)
    
    st.divider()
    st.subheader("📊 Model Benchmark")
    if metrics_summary:
        summary_df = pd.DataFrame(metrics_summary)
        st.dataframe(
            summary_df[["Model", "R2 Score", "MAE (INR)"]].rename(
                columns={"R2 Score": "R²", "MAE (INR)": "MAE (₹)"}
            ),
            hide_index=True
        )
        st.caption(f"🏆 **Active Model:** {best_model_name} (R² = {metadata.get('best_model_r2', 0.92):.2%})")
    
    st.divider()
    st.markdown("""
    **Evaluation Metrics Guide:**
    - **R² Score:** Proportion of price variance explained by features.
    - **MAE:** Average rupee difference between prediction and actual price.
    """)


# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">🚗 Used Car Resale Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enter the vehicle specifications below to estimate its fair market resale value using trained machine learning models.</div>',
    unsafe_allow_html=True
)

# Input Form
with st.form("car_prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚘 Vehicle Basics")
        brand = st.selectbox(
            "Car Brand / Make",
            options=feature_info.get("brands", ["Maruti", "Hyundai", "Tata", "Honda", "Toyota"]),
            index=feature_info.get("brands", []).index("Maruti") if "Maruti" in feature_info.get("brands", []) else 0,
            help="Select the vehicle manufacturer"
        )
        year = st.slider(
            "Manufacturing Year",
            min_value=int(feature_info.get("year_min", 2000)),
            max_value=2026,
            value=2017,
            step=1,
            help="Year when the car was manufactured"
        )
        km_driven = st.number_input(
            "Kilometers Driven (km)",
            min_value=500,
            max_value=500000,
            value=45000,
            step=1000,
            help="Total odometer reading"
        )

    with col2:
        st.markdown("### ⚙️ Fuel & Ownership")
        fuel = st.selectbox(
            "Fuel Type",
            options=feature_info.get("fuel_types", ["Diesel", "Petrol", "CNG", "LPG"]),
            index=1,
            help="Primary fuel consumed by the car"
        )
        transmission = st.radio(
            "Transmission Type",
            options=feature_info.get("transmission_types", ["Manual", "Automatic"]),
            horizontal=True,
            index=0,
            help="Manual gearbox or Automatic transmission"
        )
        seller_type = st.selectbox(
            "Seller Type",
            options=feature_info.get("seller_types", ["Individual", "Dealer", "Trustmark Dealer"]),
            index=0,
            help="Who is listing the vehicle for resale"
        )
        owner = st.selectbox(
            "Ownership History",
            options=feature_info.get("owner_types", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
            index=0,
            help="Number of previous owners"
        )

    with col3:
        st.markdown("### 🔧 Technical Specs")
        mileage = st.number_input(
            "Mileage (kmpl or km/kg)",
            min_value=5.0,
            max_value=40.0,
            value=21.4,
            step=0.5,
            help="Fuel efficiency reported by manufacturer"
        )
        engine = st.number_input(
            "Engine Displacement (CC)",
            min_value=600.0,
            max_value=5000.0,
            value=1197.0,
            step=50.0,
            help="Engine capacity in cubic centimeters"
        )
        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=25.0,
            max_value=500.0,
            value=82.0,
            step=1.0,
            help="Brake Horsepower (BHP)"
        )
        seats = st.selectbox(
            "Seating Capacity",
            options=feature_info.get("seats_options", [4, 5, 6, 7, 8]),
            index=feature_info.get("seats_options", [4, 5, 6, 7, 8]).index(5) if 5 in feature_info.get("seats_options", []) else 0,
            help="Number of passenger seats"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🔍 Calculate Estimated Resale Price", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Prediction Logic & Output
# ---------------------------------------------------------
if submit_button:
    # Build single-sample DataFrame with matching feature names
    input_data = pd.DataFrame([{
        'brand': brand,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': float(mileage),
        'engine': float(engine),
        'max_power': float(max_power),
        'seats': int(seats)
    }])
    
    with st.spinner("Analyzing market valuation through trained Random Forest pipeline..."):
        try:
            predicted_price = float(model.predict(input_data)[0])
            # Avoid unrealistic negative predictions
            predicted_price = max(25000.0, predicted_price)
            
            # Use model's benchmark MAE for confidence range
            best_mae = 74696.88
            lower_bound = max(20000.0, predicted_price - best_mae)
            upper_bound = predicted_price + best_mae
            
            # Display Prediction Card
            st.markdown(f"""
            <div class="price-box">
                <div style="font-size: 1.15rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                    Estimated Resale Value
                </div>
                <div class="price-amount">{format_inr(predicted_price)}</div>
                <div class="price-range">
                    Expected Market Range: <b>{format_inr(lower_bound)}</b> — <b>{format_inr(upper_bound)}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature impact breakdown
            st.markdown("### 📋 Vehicle Summary & Valuation Insights")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Car Age", f"{2026 - year} years", help="Calculated from manufacturing year")
            with res_col2:
                st.metric("Odometer", f"{km_driven:,} km")
            with res_col3:
                st.metric("Engine & Power", f"{int(engine)} CC | {max_power:.0f} bhp")
            with res_col4:
                st.metric("Efficiency", f"{mileage} kmpl")
                
            st.info(
                f"💡 **Model Explanation:** The predicted price is generated using **{best_model_name}** "
                f"trained on Kaggle CarDekho historical sales records. The prediction achieves a test accuracy of "
                f"**{metadata.get('best_model_r2', 0.92):.2%} R²** with an average error of approximately ±₹74,700."
            )
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("College Machine Learning Capstone Project | Dataset: Kaggle CarDekho | Built with Streamlit & Scikit-Learn")
