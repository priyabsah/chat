from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import joblib  # Import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('trained_model.pkl')  # Load your trained model here
vectorizer = joblib.load('vectorizer.pkl')  # Load your TfidfVectorizer here

# Load data for responses
data = pd.read_csv('concatenated_data.csv')  # Load your data here

# Define route for the root URL
@app.route('/')
def index():
    return 'Welcome to the Chatbot!'

# Define route for receiving user input
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    # Call function to process user input and get response
    response = process_input(user_input)
    
    return jsonify({'response': response})

# Function to process user input and generate response
def process_input(user_input):
    # Vectorize user input
    user_input_vec = vectorizer.transform([user_input.lower()])
    
    # Predict the intent
    predicted_intent = model.predict(user_input_vec)[0]
    
    # Implement response generation mechanism based on predicted intent
    if predicted_intent in data['Questions'].values:
        response = data[data['Questions'] == predicted_intent]['Answers'].values[0]
    else:
        response = "Sorry, I don't have information about this topic."
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
