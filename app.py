import streamlit as st
import numpy as np

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
)

# ---------------------------------------------------------
# CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
.big-percent{
    font-size: 42px;
    font-weight: 800;
    color: #333;
}
.metric-card{
    background: #F5F6FA;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #FF4B4B;
    font-size: 18px;
}
.stButton>button{
    width: 100%;
    height: 50px;
    background: #FF4B4B;
    color: white;
    font-size: 20px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🚚 Delivery Delay Predictor (Rule-Based)")
st.markdown("---")

# ---------------------------------------------------------
# INPUT LAYOUT
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.subheader("📝 Order Inputs")

    platform = st.selectbox(
        "Select Platform",
        ["Blinkit", "JioMart", "Swiggy Instamart", "Zepto", "Amazon Fresh", "BigBasket"]
    )

    category = st.selectbox(
        "Product Category",
        ["Dairy", "Fruits & Vegetables", "Snacks",
         "Beverages", "Personal Care", "Household", "Electronics"]
    )

    c1, c2 = st.columns(2)
    order_value = c1.number_input("Order Value (₹)", min_value=50, value=450, step=10)
    order_hour  = c2.slider("Order Hour (24h)", min_value=0, max_value=23, value=18)

# ---------------------------------------------------------
# RULE-BASED ENGINE
# ---------------------------------------------------------
def rule_based_delay(platform, category, value, hour):

    # 1. Platform Risk
    platform_risk = {
        "Blinkit": 0.10,
        "Zepto": 0.12,
        "Swiggy Instamart": 0.15,
        "Amazon Fresh": 0.20,
        "BigBasket": 0.30,
        "JioMart": 0.35
    }[platform]

    # 2. Category Risk
    category_risk = {
        "Dairy": 0.20,
        "Fruits & Vegetables": 0.18,
        "Snacks": 0.10,
        "Beverages": 0.12,
        "Personal Care": 0.08,
        "Household": 0.10,
        "Electronics": 0.25
    }[category]

    # 3. Order Value Risk
    if value < 400:
        value_risk = 0.10
    elif value <= 800:
        value_risk = 0.15
    else:
        value_risk = 0.25

    # 4. Order Hour Risk
    if 17 <= hour <= 22:
        hour_risk = 0.30
    elif 12 <= hour <= 16:
        hour_risk = 0.15
    else:
        hour_risk = 0.07

    # TOTAL RISK
    total_risk = platform_risk + category_risk + value_risk + hour_risk
    return float(np.clip(total_risk, 0, 1))

# ---------------------------------------------------------
# OUTPUT PANEL
# ---------------------------------------------------------
with col_right:
    st.subheader("📊 Delivery Status")

    if st.button("🚀 Predict Status"):

        risk_score = rule_based_delay(platform, category, order_value, order_hour)

        # Show probability
        st.markdown(f"<p class='big-percent'>{risk_score*100:.1f}%</p>", unsafe_allow_html=True)
        st.progress(risk_score)

        # Final Decision
        if risk_score > 0.50:
            st.error("### 🛑 DELIVERY STATUS: DELAY")
            st.markdown("""
                <div class='metric-card'>
                Delay is predicted due to platform, category, value & hour.
                </div>
            """, unsafe_allow_html=True)

        else:
            st.success("### ✅ DELIVERY STATUS: ON TIME")
            st.markdown("""
                <div class='metric-card' style='border-left:5px solid #00C853;'>
                Delivery is expected to be on time.
                </div>
            """, unsafe_allow_html=True)

