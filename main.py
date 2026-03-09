import fastapi as Fp
from pydantic import BaseModel # pydantic use for defining the data model for the API request and response

import joblib # joblib use for loading the pre-trained machine learning model
app = Fp.FastAPI() # create an instance of the FastAPI class
class InputData(BaseModel): # define a data model for the API request
    airline_sentiment_confidence:float
    negativereason:int
    negativereason_confidence:float
    airline:int
    retweet_count:int
    user_timezone:int
    
knn_model = joblib.load("C:\\study_material\\DA\\models\\knn_classifier") # load the pre-trained machine learning model

logistic_model = joblib.load("C:\\study_material\\DA\\models\\logistic_regression") # load the pre-trained machine learning model

standard_scaler = joblib.load("C:\\study_material\\DA\\models\\standard_scaler") # load the pre-trained standard scaler
@app.post("/knn_model") # define a POST endpoint for making predictions
def predict_knn(input_data: InputData):
    input_df = [[
        input_data.airline_sentiment_confidence,
        input_data.negativereason,
        input_data.negativereason_confidence,
        input_data.airline,
        input_data.retweet_count,
        input_data.user_timezone
    ]]
    input_df = standard_scaler.transform(input_df) # scale the input data using the pre-trained standard scaler
    prediction = knn_model.predict(input_df) # make a prediction using the pre-trained KNN model
    return {
    "model": "knn_classification",
    "prediction": "neutral" if prediction == 0 
                  else "positive" if prediction == 2 
                  else "negative"
}  # return the prediction as a JSON response

@app.post("/logistic_model") # define a POST endpoint for making predictions
def predict_logistic(input_data: InputData):
    input_df = [[
        input_data.airline_sentiment_confidence,
        input_data.negativereason,
        input_data.negativereason_confidence,
        input_data.airline,
        input_data.retweet_count,
        input_data.user_timezone
    ]]
    input_df = standard_scaler.transform(input_df) # scale the input data using the pre-trained standard scaler
    prediction = logistic_model.predict(input_df) # make a prediction using the pre-trained logistic regression model
    return {"prediction": int(prediction[0])} # return the prediction as a JSON response

