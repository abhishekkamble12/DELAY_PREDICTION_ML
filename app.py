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

# delivery_rule_with_model_import.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# IMPORTANT: uploaded file path
# (developer provided). We include it as-is.
# -----------------------------
IMAGE_URL = "/mnt/data/39b686f2-0f71-4dbc-92db-b9c0c12e4135.png"

# ---------------------------------------------------------
# 1. PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Predictor (Model Imported, Not Used)",
    page_icon="🚚",
    layout="wide",
)

# ---------------------------------------------------------
# Simple styling
# ---------------------------------------------------------
st.markdown("""
<style>
.big-font { font-size: 28px !important; font-weight: 700; }
.reason-box { background:#f7f9fb; padding:14px; border-radius:8px; border-left:6px solid #6c7ae0; }
.stButton>button { width:100%; height:46px; border-radius:8px; background:#2e86de; color:white; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. (Optional) Load model file - but DO NOT USE IT
# This demonstrates "import model but don't use".
# If file missing, continue gracefully.
# ---------------------------------------------------------
MODEL_FILENAME = "my_modelss.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

@st.cache_resource
def load_model_if_exists(path):
    """Try to load model if present. We intentionally won't use it."""
    if not os.path.exists(path):
        # Not an error — we simply note that model is absent.
        return None, f"Model not found at {path}"
    try:
        m = joblib.load(path)
        return m, "Model loaded (note: this app does NOT use the model for predictions)."
    except Exception as e:
        return None, f"Model load failed: {e}"

model, model_status = load_model_if_exists(MODEL_PATH)

# ---------------------------------------------------------
# 3. Header + show uploaded image (if exists)
# ---------------------------------------------------------
st.title("🚚 Delivery Delay Predictor — Rule-Based (Model Imported, Not Used)")
st.write(model_status)

# show the uploaded image (local path used as 'url' here)
if os.path.exists(IMAGE_URL):
    st.image(IMAGE_URL, caption="Uploaded UI screenshot (local file)", use_column_width=True)
else:
    st.caption(f"Image not found at: {IMAGE_URL} (file may not exist in this environment)")

st.markdown("---")

# ---------------------------------------------------------
# 4. Inputs
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("📝 Order Details")
    platform = st.selectbox("Platform", ["Blinkit", "JioMart", "Swiggy Instamart", "Zepto", "Amazon Fresh", "BigBasket"])
    product_category = st.selectbox("Product Category",
                                    ["Dairy", "Fruits & Vegetables", "Snacks", "Beverages", "Personal Care", "Household", "Electronics"])
    c1, c2 = st.columns(2)
    order_value = c1.number_input("Order Value (INR)", min_value=10, value=350, step=10)
    order_hour = c2.slider("Order Hour (24h)", 0, 23, 18)
    

# ---------------------------------------------------------
# 5. Rule-based delay logic (final decision depends on
#     platform, product category, order value, order hour)
# ---------------------------------------------------------
def compute_rule_based_risk(platform, category, value, hour):
    # Platform risk weights (higher -> more likely delay)
    platform_risk_map = {
        "Blinkit": 0.04,
        "Zepto": 0.05,
        "Swiggy Instamart": 0.06,
        "Amazon Fresh": 0.08,
        "BigBasket": 0.12,
        "JioMart": 0.15
    }
    risk_platform = platform_risk_map.get(platform, 0.05)

    # Category risk weights
    category_risk_map = {
        "Dairy": 0.10,
        "Fruits & Vegetables": 0.08,
        "Snacks": 0.04,
        "Beverages": 0.05,
        "Personal Care": 0.02,
        "Household": 0.03,
        "Electronics": 0.12
    }
    risk_category = category_risk_map.get(category, 0.03)

    # Order value impact
    if value < 400:
        risk_value = 0.02
    elif value <= 800:
        risk_value = 0.05
    else:
        risk_value = 0.10

    # Hour impact
    if 16 <= hour <= 21:
        risk_hour = 0.15
    elif 11 <= hour <= 15:
        risk_hour = 0.07
    else:
        risk_hour = 0.03

    # Combine
    total = risk_platform + risk_category + risk_value + risk_hour

    # Optionally apply a mild non-linear dampening so extreme sums don't always exceed 1
    # but here we just clip between 0 and 1
    total = float(np.clip(total, 0.0, 1.0))
    breakdown = {
        "platform": risk_platform,
        "category": risk_category,
        "value": risk_value,
        "hour": risk_hour,
        "total": total
    }
    return breakdown

# ---------------------------------------------------------
# 6. Output
# ---------------------------------------------------------
with col_right:
    st.subheader("🧠 Prediction (Rule-Based)")
    st.write("")
    if st.button("Predict Delivery Status (rule-based)"):
        bd = compute_rule_based_risk(platform, product_category, order_value, order_hour)
        final_risk = bd["total"]

        st.markdown(f"<p class='big-font'>Delay Probability: {final_risk:.1%}</p>", unsafe_allow_html=True)
        st.progress(final_risk)

        # decision threshold (you can tweak this)
        threshold = 0.50
        if final_risk > threshold:
            st.error("### 🛑 Delivery Status: HIGH RISK OF DELAY")
            st.markdown(f"""
                <div class="reason-box">
                    <b>Breakdown:</b><br>
                    • Platform risk: {bd['platform']*100:.1f}%<br>
                    • Category risk: {bd['category']*100:.1f}%<br>
                    • Order value risk: {bd['value']*100:.1f}%<br>
                    • Order hour risk: {bd['hour']*100:.1f}%<br><br>
                    <b>Action:</b> Notify customer with ETA buffer (10–20 mins).
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("### ✅ Delivery Status: ON TIME")
            st.markdown(f"""
                <div class="reason-box" style="border-left:6px solid #00b894;">
                    <b>Breakdown:</b><br>
                    • Platform risk: {bd['platform']*100:.1f}%<br>
                    • Category risk: {bd['category']*100:.1f}%<br>
                    • Order value risk: {bd['value']*100:.1f}%<br>
                    • Order hour risk: {bd['hour']*100:.1f}%<br><br>
                    <b>Action:</b> Proceed with normal dispatch.
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer - developer note
# ---------------------------------------------------------
st.markdown("---")
# st.caption("Developer note: Model file is loaded (if present) but intentionally not used. The prediction is fully rule-based.")
