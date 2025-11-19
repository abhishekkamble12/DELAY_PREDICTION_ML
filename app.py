import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .main-header {
        font-size: 2.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load the Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Try loading the specific file the user mentioned
    filename = 'my_model.pkl'
    
    # Fallback if they renamed it or used the other script
    if not os.path.exists(filename):
        filename = 'best_delivery_model.pkl'
        
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

model = load_model()

# ---------------------------------------------------------
# 3. User Interface
# ---------------------------------------------------------
st.markdown("<h1 class='main-header'>🚚 Delivery Delay Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the order details below to check if the delivery is likely to be **Delayed** or **On Time**.")

if model is None:
    st.error("⚠️ **Model not found!** Please make sure 'my_model.pkl' is in the same folder as this script.")
else:
    # Create a form for inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Order Details")
            
            # Platform Options (Based on your dataset)
            platform = st.selectbox(
                "Platform",
                ["Blinkit", "JioMart", "Swiggy Instamart", "Zepto", "Amazon Fresh", "BigBasket"]
            )
            
            # Product Category Options
            category = st.selectbox(
                "Product Category",
                ["Dairy", "Fruits & Vegetables", "Snacks", "Beverages", "Personal Care", "Household", "Electronics"]
            )
            
            order_value = st.number_input(
                "Order Value (INR)", 
                min_value=50, 
                max_value=50000, 
                value=450,
                step=50
            )

        with col2:
            st.subheader("🕒 Timing & Service")
            
            # Order Hour (0 - 23)
            order_hour = st.slider(
                "Order Hour (24h format)", 
                min_value=0, 
                max_value=23, 
                value=18,
                help="Example: 18 means 6 PM"
            )
            
            # Service Rating
            # Note: This was a feature in your training data
            rating = st.slider(
                "Expected Service Rating (1-5)", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="1 = Poor, 5 = Excellent"
            )

        submit_button = st.form_submit_button("🚀 Predict Status")

    # ---------------------------------------------------------
    # 4. Prediction Logic
    # ---------------------------------------------------------
    if submit_button:
        # Prepare input data matching the training columns exactly
        input_data = pd.DataFrame({
            'Platform': [platform],
            'Product Category': [category],
            'Order Value (INR)': [order_value],
            'Service Rating': [rating],
            'Order_Hour': [order_hour]
        })

        try:
            # Make Prediction
            prediction = model.predict(input_data)
            
            # Display Result
            st.markdown("---")
            if prediction[0] == 1:
                st.error("### ⚠️ Prediction: DELAY LIKELY")
                st.write("This order has a high risk of being delayed based on historical patterns.")
            else:
                st.success("### ✅ Prediction: ON TIME")
                st.write("This order is expected to arrive on time.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Tip: Ensure the input features match exactly what the model was trained on.")