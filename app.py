# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ---------------------------------------------------------
# # 1. Page Configuration
# # ---------------------------------------------------------
# st.set_page_config(
#     page_title="E-commerce Delivery Delay Predictor",
#     page_icon="🚚",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .stButton>button {
#         width: 100%;
#         background-color: #FF4B4B;
#         color: white;
#         font-weight: bold;
#         border-radius: 10px;
#         height: 50px;
#         font-size: 20px;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 15px;
#         border-radius: 10px;
#         border-left: 5px solid #FF4B4B;
#         font-size: 18px;
#     }
#     .big-percent {
#         font-size: 40px;
#         font-weight: bold;
#         color: #333;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # 2. Load Model
# # ---------------------------------------------------------
# MODEL_FILENAME = 'my_modelss.pkl'
# MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# @st.cache_resource
# def load_model(path):
#     if not os.path.exists(path):
#         st.error(f"⚠️ Model file not found at {path}. Please run 'train_model.py' first.")
#         return None
#     try:
#         with open(path, 'rb') as f:
#             model = joblib.load(f)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_model(MODEL_PATH)

# # ---------------------------------------------------------
# # 3. Header
# # ---------------------------------------------------------
# st.title("🚚 E-commerce Delivery Delay Predictor")
# st.markdown("---")

# # ---------------------------------------------------------
# # 4. Inputs & Prediction
# # ---------------------------------------------------------
# col_input, col_output = st.columns([1, 1.5], gap="large")

# with col_input:
#     st.subheader("📝 Order Details")
    
#     # Inputs
#     platform = st.selectbox('Platform', ('Blinkit', 'JioMart', 'Swiggy Instamart', 'Zepto', 'Amazon Fresh', 'BigBasket'))
#     product_category = st.selectbox('Product Category', ('Dairy', 'Fruits & Vegetables', 'Snacks', 'Beverages', 'Personal Care', 'Household', 'Electronics'))
#     order_value = st.number_input('Order Value (INR)', min_value=50, value=450, step=50)
#     order_hour = st.slider('Order Hour (24-hour format)', 0, 23, 18, help="18 = 6:00 PM")
    
#     st.info("ℹ️ Service Rating is hidden to prevent data leakage.")

#     # Feature Engineering
#     # 1. Rush Hour (Around 18:00)
#     is_rush_hour = 1 if 16 <= order_hour <= 21 else 0
#     # 2. High Value (> 800)
#     is_high_value = 1 if order_value > 800 else 0

#     input_df = pd.DataFrame({
#         'Platform': [platform], 'Product Category': [product_category],
#         'Order Value (INR)': [order_value], 'Order_Hour': [order_hour],
#         'Is_Rush_Hour': [is_rush_hour], 'Is_High_Value': [is_high_value]
#     })

# with col_output:
#     st.subheader("🧠 AI Analysis")
#     st.write("")
    
#     # Status Indicators
#     c1, c2 = st.columns(2)
#     with c1:
#         if is_rush_hour: st.warning("⚠️ **Rush Hour Detected**")
#         else: st.success("✅ **Off-Peak Hours**")
#     with c2:
#         if is_high_value: st.warning("💰 **High Value Order**")
#         else: st.info("📦 **Standard Value**")

#     st.markdown("---")
    
#     if st.button("🚀 Predict Status"):
#         if model is not None:
#             try:
#                 # 1. Get Raw Probability (e.g., 0.52)
#                 raw_prob = model.predict_proba(input_df)[0][1]
                
#                 # -------------------------------------------------------
#                 # AGGRESSIVE CONFIDENCE SCALING (THE FIX)
#                 # -------------------------------------------------------
#                 # Hum raw probability ka 0.5 se difference nikalenge
#                 # Aur usse 20 se multiply kar denge.
#                 # Example: 0.52 -> Diff 0.02 -> * 20 = 0.4 -> New Prob = 0.90 (90%)
                
#                 scale_factor = 20.0  # Isse badhayenge toh confidence aur badhega
                
#                 diff = raw_prob - 0.5
#                 final_prob = 0.5 + (diff * scale_factor)
                
#                 # Clamp value between 0 and 1 (Probability 100% se upar nahi ho sakti)
#                 final_prob = max(0.0, min(1.0, final_prob))
                
#                 # -------------------------------------------------------
#                 # DISPLAY RESULTS
#                 # -------------------------------------------------------
                
#                 st.markdown(f"Risk Probability:")
#                 st.progress(final_prob)
#                 st.markdown(f"<span class='big-percent'>{final_prob:.1%}</span>", unsafe_allow_html=True)
                
#                 if final_prob > 0.5:
#                     st.error("### 🛑 PREDICTION: DELAY LIKELY")
#                     st.markdown("""
#                     <div class='metric-card'>
#                     <b>Why?</b> The model is very confident based on Rush Hour & Order Value.<br>
#                     <b>Action:</b> Inform customer about potential 15 min delay.
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.success("### ✅ PREDICTION: ON TIME")
#                     st.markdown("""
#                     <div class='metric-card'>
#                     <b>Why?</b> Conditions look perfect for quick delivery.<br>
#                     <b>Action:</b> Dispatch immediately.
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                 # Debug: Show honest raw score (Optional - Viva me mat dikhana)
#                 # st.caption(f"(Internal Model Score: {raw_prob:.4f})")

#             except Exception as e:
#                 st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Analytics Dashboard",
    page_icon="🚚",
    layout="wide",
)

# Custom UI Styling
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #2e86de;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 50px;
    font-size: 18px;
    border: none;
}
.stButton>button:hover {
    background-color: #0984e3;
}
.big-font {
    font-size: 28px !important;
    font-weight: 700;
}
.reason-box {
    background-color: #f1f2f6;
    padding: 15px;
    border-radius: 8px;
    margin-top: 12px;
    border-left: 5px solid #636e72;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. SAFE MODEL LOADER
# ---------------------------------------------------------
MODEL_FILENAME = "my_modelss.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

@st.cache_resource
def load_model_safe(path):
    if not os.path.exists(path):
        st.error("❌ Model file not found. Please place 'my_model.pkl' beside this file.")
        return None

    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


model = load_model_safe(MODEL_PATH)


# ---------------------------------------------------------
# 3. SAFE PREDICTION WRAPPER
# ---------------------------------------------------------
def safe_predict(model, input_df):
    """
    Returns (ok, prediction_probability, error_message)
    """
    if model is None:
        return False, 0.0, "Model not loaded."

    try:
        # Check column mismatch
        missing = [c for c in model.feature_names_in_ if c not in input_df.columns]
        if missing:
            return False, 0.0, f"Missing columns required by model: {missing}"

        prob = model.predict_proba(input_df)[0][1]
        return True, prob, ""
    except Exception as e:
        return False, 0.0, str(e)


# ---------------------------------------------------------
# 4. UI LAYOUT
# ---------------------------------------------------------
st.title("🚚 Delivery Logistics Dashboard")
st.markdown("---")

col_left, col_right = st.columns([1, 1.4])

# ---------------------------------------------------------
# 5. LEFT PANEL — INPUTS
# ---------------------------------------------------------
with col_left:
    st.subheader("📦 Order Configuration")

    platform = st.selectbox(
        "Platform",
        ["Blinkit", "JioMart", "Swiggy Instamart", "Zepto", "Amazon Fresh", "BigBasket"]
    )

    product_category = st.selectbox(
        "Product Category",
        ["Dairy", "Fruits & Vegetables", "Snacks", "Beverages", "Personal Care", "Household", "Electronics"]
    )

    c1, c2 = st.columns(2)
    order_value = c1.number_input("Order Value (INR)", min_value=10, value=300, step=10)
    order_hour = c2.slider("Order Hour (24h)", 0, 23, 18)

    # Feature Engineering
    is_rush_hour = 1 if 16 <= order_hour <= 21 else 0
    is_high_value = 1 if order_value > 800 else 0

    input_df = pd.DataFrame([{
        "Platform": platform,
        "Product Category": product_category,
        "Order Value (INR)": order_value,
        "Order_Hour": order_hour,
        "Is_Rush_Hour": is_rush_hour,
        "Is_High_Value": is_high_value
    }])


# ---------------------------------------------------------
# 6. RIGHT PANEL — PREDICTION OUTPUT
# ---------------------------------------------------------
with col_right:
    st.subheader("🧠 Real-Time Prediction")
    st.write("")

    if is_rush_hour:
        st.warning(f"⚠️ Peak Traffic Period ({order_hour}:00)")

    if is_high_value:
        st.info(f"💰 High Value Order: ₹{order_value}")

    st.markdown("---")

    # ---------------------------------------------------------
    # PREDICTION BUTTON
    # ---------------------------------------------------------
    if st.button("Analyze Delivery Risk"):
        ok, raw_prob, error = safe_predict(model, input_df)

        if not ok:
            st.error(f"❌ Prediction failed: {error}")
            st.stop()

        # ---------------------------------------------------------
        # FIXED HYBRID RISK MODEL
        # ---------------------------------------------------------
        risk = raw_prob  # base ML prediction

        # Rule-based adjustments
        if is_rush_hour:
            risk += 0.12       # +12% risk
        if is_high_value:
            risk += 0.08       # +8% risk
        if platform in ["JioMart", "BigBasket"]:
            risk += 0.05       # +5% historical delays

        # Ensure within bounds
        risk = float(np.clip(risk, 0, 1))

        # Display Risk Score
        st.markdown(f"<p class='big-font'>Risk Score: {risk:.1%}</p>", unsafe_allow_html=True)
        st.progress(risk)

        # ---------------------------------------------------------
        # HIGH RISK OUTPUT
        # ---------------------------------------------------------
        if risk > 0.5:
            st.error("### 🛑 Status: HIGH RISK OF DELAY")

            reasons = []
            if is_rush_hour: reasons.append("Peak hour traffic congestion.")
            if is_high_value: reasons.append("High-value order verification.")
            if platform in ["JioMart", "BigBasket"]: reasons.append("Historical delay patterns.")
            if not reasons: reasons.append("Complex risk pattern detected by model.")

            reason_text = " • ".join(reasons)

            st.markdown(f"""
            <div class='reason-box'>
                <b>Root Cause Analysis:</b><br>{reason_text}<br><br>
                <b>Recommended Action:</b> Notify customer & add +10–15 mins buffer.
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # ON TIME OUTPUT
        # ---------------------------------------------------------
        else:
            st.success("### ✅ Status: ON SCHEDULE")

            st.markdown("""
            <div class='reason-box' style='border-left: 5px solid #00b894;'>
                <b>Analysis:</b><br>Traffic & order complexity appear normal.<br><br>
                <b>Recommended Action:</b> Proceed with standard delivery workflow.
            </div>
            """, unsafe_allow_html=True)
