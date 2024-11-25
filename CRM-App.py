import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Helper Functions
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = text.lower()
        return text
    return ''

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load datasets
@st.cache
def load_data():
    train_df = pd.read_csv("twitter_training.csv")
    val_df = pd.read_csv("twitter_validation.csv")
    tweets_df = pd.read_csv("Twitter Scraping Tweets Dataset.csv")
    return train_df, val_df, tweets_df

# Preprocess data
def preprocess_data(df):
    df['cleaned_text'] = df.iloc[:, 3].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)
    return df

# Streamlit App
st.title("Customer Sentiment Analysis Web App")
st.write("""
This app allows you to:
1. **Upload datasets** for training and validation.
2. **Clean, preprocess, and visualize data**.
3. **Train and evaluate a sentiment analysis model**.
4. **Predict sentiment on new data**.
""")

# Sidebar for file upload
st.sidebar.header("Upload Options")
train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
val_file = st.sidebar.file_uploader("Upload Validation Data (CSV)", type=["csv"])
tweets_file = st.sidebar.file_uploader("Upload Twitter Dataset (CSV)", type=["csv"])

if train_file and val_file and tweets_file:
    # Load uploaded files
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    tweets_df = pd.read_csv(tweets_file)
    
    # Display sample data
    st.write("### Sample Data from Training Dataset")
    st.write(train_df.head())

    # Preprocess data
    st.write("### Preprocessing Data...")
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    tweets_df = preprocess_data(tweets_df)

    # TF-IDF Vectorization
    st.write("### Vectorizing Text Data...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['cleaned_text'])
    X_val = tfidf.transform(val_df['cleaned_text'])
    X_tweets = tfidf.transform(tweets_df['cleaned_text'])
    y_train = train_df.iloc[:, 2]
    y_val = val_df.iloc[:, 2]

    # Model Training
    st.write("### Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    st.write(f"Model Accuracy: **{accuracy:.2f}**")

    # Prediction on Tweets Dataset
    tweets_df['predicted_sentiment'] = model.predict(X_tweets)
    st.write("### Predictions on New Tweets")
    st.write(tweets_df[['user_name', 'text', 'predicted_sentiment']].head())

    # Visualization
    st.write("### Sentiment Distribution")
    sentiment_counts = tweets_df['predicted_sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

else:
    st.write("Please upload all required files to proceed.")