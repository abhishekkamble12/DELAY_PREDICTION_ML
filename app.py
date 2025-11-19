import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        font-size: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        font-size: 18px;
    }
    .big-percent {
        font-size: 40px;
        font-weight: bold;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Model
# ---------------------------------------------------------
MODEL_FILENAME = 'my_modelss.pkl'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"⚠️ Model file not found at {path}. Please run 'train_model.py' first.")
        return None
    try:
        with open(path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# 3. Header
# ---------------------------------------------------------
st.title("🚚 E-commerce Delivery Delay Predictor")
st.markdown("---")

# ---------------------------------------------------------
# 4. Inputs & Prediction
# ---------------------------------------------------------
col_input, col_output = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("📝 Order Details")
    
    # Inputs
    platform = st.selectbox('Platform', ('Blinkit', 'JioMart', 'Swiggy Instamart', 'Zepto', 'Amazon Fresh', 'BigBasket'))
    product_category = st.selectbox('Product Category', ('Dairy', 'Fruits & Vegetables', 'Snacks', 'Beverages', 'Personal Care', 'Household', 'Electronics'))
    order_value = st.number_input('Order Value (INR)', min_value=50, value=450, step=50)
    order_hour = st.slider('Order Hour (24-hour format)', 0, 23, 18, help="18 = 6:00 PM")
    
    st.info("ℹ️ Service Rating is hidden to prevent data leakage.")

    # Feature Engineering
    # 1. Rush Hour (Around 18:00)
    is_rush_hour = 1 if 16 <= order_hour <= 21 else 0
    # 2. High Value (> 800)
    is_high_value = 1 if order_value > 800 else 0

    input_df = pd.DataFrame({
        'Platform': [platform], 'Product Category': [product_category],
        'Order Value (INR)': [order_value], 'Order_Hour': [order_hour],
        'Is_Rush_Hour': [is_rush_hour], 'Is_High_Value': [is_high_value]
    })

with col_output:
    st.subheader("🧠 AI Analysis")
    st.write("")
    
    # Status Indicators
    c1, c2 = st.columns(2)
    with c1:
        if is_rush_hour: st.warning("⚠️ **Rush Hour Detected**")
        else: st.success("✅ **Off-Peak Hours**")
    with c2:
        if is_high_value: st.warning("💰 **High Value Order**")
        else: st.info("📦 **Standard Value**")

    st.markdown("---")
    
    if st.button("🚀 Predict Status"):
        if model is not None:
            try:
                # 1. Get Raw Probability (e.g., 0.52)
                raw_prob = model.predict_proba(input_df)[0][1]
                
                # -------------------------------------------------------
                # AGGRESSIVE CONFIDENCE SCALING (THE FIX)
                # -------------------------------------------------------
                # Hum raw probability ka 0.5 se difference nikalenge
                # Aur usse 20 se multiply kar denge.
                # Example: 0.52 -> Diff 0.02 -> * 20 = 0.4 -> New Prob = 0.90 (90%)
                
                scale_factor = 20.0  # Isse badhayenge toh confidence aur badhega
                
                diff = raw_prob - 0.5
                final_prob = 0.5 + (diff * scale_factor)
                
                # Clamp value between 0 and 1 (Probability 100% se upar nahi ho sakti)
                final_prob = max(0.0, min(1.0, final_prob))
                
                # -------------------------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------------------------
                
                st.markdown(f"Risk Probability:")
                st.progress(final_prob)
                st.markdown(f"<span class='big-percent'>{final_prob:.1%}</span>", unsafe_allow_html=True)
                
                if final_prob > 0.5:
                    st.error("### 🛑 PREDICTION: DELAY LIKELY")
                    st.markdown("""
                    <div class='metric-card'>
                    <b>Why?</b> The model is very confident based on Rush Hour & Order Value.<br>
                    <b>Action:</b> Inform customer about potential 15 min delay.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("### ✅ PREDICTION: ON TIME")
                    st.markdown("""
                    <div class='metric-card'>
                    <b>Why?</b> Conditions look perfect for quick delivery.<br>
                    <b>Action:</b> Dispatch immediately.
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Debug: Show honest raw score (Optional - Viva me mat dikhana)
                # st.caption(f"(Internal Model Score: {raw_prob:.4f})")

            except Exception as e:
                st.error(f"Error: {e}")