import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

def train_and_save_model():
    data = pd.read_csv('sentiment_dataset.csv')

    X = data['text']
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(CountVectorizer(), SVC(kernel='linear'))

    model.fit(X_train, y_train)

    joblib.dump(model, 'sentiment_model.pkl')
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    train_and_save_model()
