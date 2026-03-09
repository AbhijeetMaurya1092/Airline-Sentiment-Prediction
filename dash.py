import requests
import streamlit as st

st.set_page_config(page_title="Airline Sentiment Predictor", page_icon=":airplane:", layout="wide")

st.title("Airline Sentiment Predictor")
st.caption("Interactive frontend for your FastAPI sentiment models")

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")
    model_choice = st.selectbox(
        "Choose Model",
        options=["KNN Classifier", "Logistic Regression"],
        index=0,
    )

    endpoint = "/knn_model" if model_choice == "KNN Classifier" else "/logistic_model"
    st.markdown(f"**Endpoint:** `{endpoint}`")

st.subheader("Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    airline_sentiment_confidence = st.number_input(
        "airline_sentiment_confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        format="%.2f",
    )
    negativereason = st.number_input("negativereason (encoded int)", min_value=0, value=0, step=1)

with col2:
    negativereason_confidence = st.number_input(
        "negativereason_confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        format="%.2f",
    )
    airline = st.number_input("airline (encoded int)", min_value=0, value=0, step=1)

with col3:
    retweet_count = st.number_input("retweet_count", min_value=0, value=0, step=1)
    user_timezone = st.number_input("user_timezone (encoded int)", value=0, step=1)

payload = {
    "airline_sentiment_confidence": float(airline_sentiment_confidence),
    "negativereason": int(negativereason),
    "negativereason_confidence": float(negativereason_confidence),
    "airline": int(airline),
    "retweet_count": int(retweet_count),
    "user_timezone": int(user_timezone),
}

if st.button("Predict", type="primary", use_container_width=True):
    url = f"{base_url.rstrip('/')}{endpoint}"

    try:
        with st.spinner("Requesting prediction from API..."):
            response = requests.post(url, json=payload, timeout=15)

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            data = response.json()
            st.success("Prediction received")

            if endpoint == "/logistic_model":
                label_map = {0: "neutral", 1: "negative", 2: "positive"}
                pred_num = data.get("prediction")
                pred_label = label_map.get(pred_num, f"unknown ({pred_num})")

                c1, c2 = st.columns(2)
                c1.metric("Predicted Class (int)", pred_num)
                c2.metric("Predicted Sentiment", pred_label)
            else:
                st.metric("Predicted Sentiment", data.get("prediction", "N/A"))

            with st.expander("Request/Response Debug"):
                st.write("Request payload:")
                st.json(payload)
                st.write("Response JSON:")
                st.json(data)

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to API at {url}")
        st.exception(exc)

st.markdown("---")
st.markdown(
    "Run backend first, then run this app. If API runs on another host/port, update it in sidebar."
)
