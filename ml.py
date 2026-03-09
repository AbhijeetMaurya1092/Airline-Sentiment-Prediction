from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import joblib


df=pd.read_csv("Tweets.csv")
print(df.info())
le =LabelEncoder()
column=["airline_sentiment", "airline", "user_timezone", "negativereason"]

for col in column:
    df[col]= le.fit_transform(df[col].astype(str))

df["negativereason"] = df["negativereason"].fillna("None")
df["negativereason_confidence"] = df["negativereason_confidence"].fillna(df["negativereason_confidence"].mode()[0])

to_be_scaled=["airline", "user_timezone", "negativereason", "negativereason_confidence","retweet_count","airline_sentiment_confidence"]
sc=StandardScaler()
df[to_be_scaled] = sc.fit_transform(df[to_be_scaled])
x=df.drop(columns=[
           "tweet_id",
           "airline_sentiment",
           "airline_sentiment_gold",
           "name",
           "text",
           "tweet_coord",
           "tweet_created",
           "tweet_location",
           "negativereason_gold",
           ],errors='ignore')

y=df["airline_sentiment"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42, test_size=0.2)

#Logistic regression
log_regg= LogisticRegression(max_iter=200)
log_regg.fit(x_train,y_train)
print("Logistic regression model trained")

#KNN model
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print("KNN classifier model trained")

log_predict = log_regg.predict(x_test)
knn_predict = knn.predict(x_test)
print("Prediction of logistic regression and KNN classification is done!")

#Evaluate the models
log_reg_confusion = confusion_matrix(y_test,log_predict)
log_class_report = classification_report(y_test,log_predict)
log_accuracy = accuracy_score(y_test,log_predict)
print(f" logistic regression confusion matric: {log_reg_confusion}\n classification report: {log_class_report}\n accuracy: {log_accuracy}")

knn_confusion = confusion_matrix(y_test,knn_predict)
knn_class_report = classification_report(y_test,knn_predict)
knn_accuracy = accuracy_score(y_test,knn_predict)
print(f" KNN confusion matrix: {knn_confusion}\n classification report: {knn_class_report}\n accuracy: {knn_accuracy}")


joblib.dump(log_regg,"models/logistic_regression")
joblib.dump(knn,"models/knn_classifier")
joblib.dump(sc,"models/standard_scaler")
print("Models and scaler saved successfully!")

