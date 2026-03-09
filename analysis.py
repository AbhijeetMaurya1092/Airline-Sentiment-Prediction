# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load Dataset
df = pd.read_csv('Tweets.csv', encoding='latin1')


# Display First Few Rows
print(df.head())


# Dataset Information
print(df.info())


# Statistical Summary
print(df.describe())


# -----------------------------
# Sentiment Distribution Graph
# -----------------------------
sns.countplot(x='airline_sentiment', data=df)
plt.title("Distribution of Airline Sentiment")
plt.show()


# -----------------------------
# Negative Reasons Distribution
# -----------------------------
sns.countplot(y='negativereason', data=df)
plt.title("Reasons for Negative Sentiment")
plt.show()


# -----------------------------
# Sentiment Confidence Histogram
# -----------------------------
plt.hist(df['airline_sentiment_confidence'], bins=20)
plt.title("Sentiment Confidence Distribution")
plt.show()


# -----------------------------
# Create Tweet Length Feature
# -----------------------------
df['tweet_length'] = df['text'].apply(len)


# -----------------------------
# Tweet Length vs Sentiment
# -----------------------------
sns.boxplot(x='airline_sentiment', y='tweet_length', data=df)
plt.title("Tweet Length vs Sentiment")
plt.show()


# -----------------------------
# Tweets Per Airline
# -----------------------------
sns.countplot(x='airline', data=df)
plt.title("Number of Tweets per Airline")
plt.xticks(rotation=45)
plt.show()


# # -----------------------------
# # Dataset Structure Check
# # -----------------------------
# df.info()


# # -----------------------------
# # Missing Values Check
# # -----------------------------
# df.isnull().sum()


# # -----------------------------
# # Value Counts
# # -----------------------------
# df['airline'].value_counts()
# df['airline_sentiment'].value_counts()

# nltk.download('stopwords')
# nltk.download('wordnet')

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
    
#     text = text.lower()   # lowercase
    
#     text = re.sub(r'http\S+', '', text)   # remove links
    
#     text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove numbers & punctuation
    
#     words = text.split()
    
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
#     return " ".join(words)


# df['clean_text'] = df['text'].apply(clean_text)

# print(df[['text','clean_text']].head())

# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(max_features=5000)

# X = vectorizer.fit_transform(df['clean_text'])

# y = df['airline_sentiment']

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )



# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(max_features=5000)

# X = vectorizer.fit_transform(df['clean_text'])

# y = df['airline_sentiment']

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()

# model.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, classification_report

# print("Accuracy:", accuracy_score(y_test, y_pred))

# print(classification_report(y_test, y_pred))


# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


# sample_tweet = ["The flight service was amazing and staff was very helpful"]
# sample_vec = vectorizer.transform(sample_tweet)
# prediction = model.predict(sample_vec)
# print("Predicted Sentiment:", prediction[0])
