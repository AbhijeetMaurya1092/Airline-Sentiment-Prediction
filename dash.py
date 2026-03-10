import streamlit as st
import requests

# Set your FastAPI backend URL here
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Airline Sentiment Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Airline Sentiment Predictor")
st.markdown("Enter the tweet and user details to predict the sentiment.")

# Model Selection
model_choice = st.radio("Choose Prediction Model:", ("KNN Classifier", "Logistic Regression"))

st.subheader("Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    airline_sentiment_confidence = st.number_input("Airline Sentiment Confidence", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    negativereason = st.number_input("Negative Reason (Encoded Integer)", value=0, step=1)
    negativereason_confidence = st.number_input("Negative Reason Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

with col2:
    airline = st.number_input("Airline (Encoded Integer)", value=1, step=1)
    retweet_count = st.number_input("Retweet Count", min_value=0, value=0, step=1)
    user_timezone = st.number_input("User Timezone (Encoded Integer)", value=0, step=1)

# Predict Button
if st.button("Predict Sentiment", type="primary"):
    # Map input data to the Pydantic model structure
    payload = {
        "airline_sentiment_confidence": airline_sentiment_confidence,
        "negativereason": negativereason,
        "negativereason_confidence": negativereason_confidence,
        "airline": airline,
        "retweet_count": retweet_count,
        "user_timezone": user_timezone
    }
    
    # Select endpoint based on user choice
    endpoint = "/knn_model" if model_choice == "KNN Classifier" else "/logistic_model"
    
    try:
        with st.spinner('Calling API...'):
            response = requests.post(f"{API_URL}{endpoint}", json=payload)
            response.raise_for_status() # Check for HTTP errors
            
            data = response.json()
            
            # Extract result based on the API response structure
            if model_choice == "KNN Classifier":
                result = data.get("knn_classification", {}).get("prediction", "Unknown")
            else:
                result = data.get("Logistic _regression", {}).get("prediction", "Unknown")
                
            st.success(f"**Predicted Sentiment:** {result.upper()}")
            st.json(data) # Show the raw JSON response for debugging

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Is your FastAPI server running on http://127.0.0.1:8000 ?")
    except Exception as e:
        st.error(f"An error occurred: {e}")