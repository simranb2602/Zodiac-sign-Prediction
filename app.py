from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import webbrowser


app = Flask(__name__)

# Load the trained model
model_path = "best_svc_model.pkl"
with open(model_path, "rb") as model_file:
    best_svc_model = pickle.load(model_file)

# Load other resources needed for preprocessing
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words=stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']

        # Preprocess the text
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words("english")]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        preprocessed_text = ' '.join(lemmatized_words)

        # Transform the preprocessed text using the loaded TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([preprocessed_text])

        # Make predictions using the model
        predicted_sign = best_svc_model.predict(transformed_text)[0]

        response = {'predicted_sign': predicted_sign}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

    # Automatically open the browser
    webbrowser.open_new('http://localhost:5000')