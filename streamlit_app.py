import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in word_tokenize(text) if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load the saved LabelEncoder
loaded_label_encoder = joblib.load('label_encoder.pkl')

additional_classes = ['Aquarius', 'Pisces', 'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn']
loaded_label_encoder.classes_ = np.concatenate((loaded_label_encoder.classes_, additional_classes))

# Load the tuned LinearSVC model
model_filename = 'tuned_linearsvc_model.pkl'
loaded_model = joblib.load(model_filename)

# Load the TF-IDF vectorizer (if you saved it)
vectorizer_filename = 'tfidf_vectorizer.pkl'
tfidf = joblib.load(vectorizer_filename)

# Streamlit app title
st.title('Zodiac Sign Prediction App')

# Input text box for the user to enter text
user_input = st.text_area('Enter some text:', '')

# Make a prediction when the user clicks the "Predict" button
if st.button('Predict'):
    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)

        # Transform the preprocessed input using the loaded TF-IDF vectorizer
        processed_input_vector = tfidf.transform([processed_input])

        # Make a prediction using the loaded model
        prediction_id = loaded_model.predict(processed_input_vector)[0]  # Get the predicted label ID
        

        print("Predicted Label ID:", prediction_id)
        print("Loaded Label Encoder Classes:", loaded_label_encoder.classes_)


        # Check if the predicted label is present in the label encoder classes
        if prediction_id in loaded_label_encoder.classes_:
            predicted_zodiac_sign = loaded_label_encoder.inverse_transform([prediction_id])[0]
            st.success(f'Predicted Zodiac Sign: {predicted_zodiac_sign}')
        else:
            st.warning('Sorry, we don\'t have enough information to predict the zodiac sign for this text.')

    else:
        st.warning('Please enter some text for prediction.')

