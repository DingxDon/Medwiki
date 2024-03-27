from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dataset
try:
    data = pd.read_json("1.format.json")
except FileNotFoundError:
    print("Dataset file not found. Please ensure '1.format.json' exists in the correct location.")
    data = None

if data is not None:
    # Preprocess text data
    for col in data.columns:
        if col != 'name':
            data[col] = data[col].astype(str).apply(lambda x: x.lower())

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit vectorizer on preprocessed text data
    tfidf_matrix = vectorizer.fit_transform(data['medicine_description'] + " " + data['medicine_generic_name'] + " " + data['dosage_form'] + " " + data['strength'] + " " + data['indication_check'] + " " + data['contraindications'] + " " + data['side_effects'] + " " + data['warnings_precautions'] + " " + data['storage_conditions'] )
else:
    vectorizer = None
    tfidf_matrix = None

def get_responses(user_query, top_n=5):
    if vectorizer is None or tfidf_matrix is None:
        return []  # Return empty list if data is not loaded properly

    # Preprocess user query
    processed_query = user_query.lower()
    
    # Transform user query into TF-IDF vector
    query_vector = vectorizer.transform([processed_query])
    
    # Calculate cosine similarity between user query vector and dataset vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    # Get indices of top N rows with highest similarity scores
    top_indices = similarity_scores.argsort(axis=1)[0][-top_n:][::-1]
    
    responses = []
    for index in top_indices:
        # Get information from the matched row
        matched_row = data.iloc[index]
        
        # Construct response
        response = {"medicine": matched_row['name'], "similarity_score": similarity_scores[0][index]}
        
        responses.append(response)
    
    return responses

@app.route('/get_responses', methods=['POST'])
def process_query():
    try:
        user_query = request.json['user_query']
    except KeyError:
        return jsonify({"error": "Please provide a user query."}), 400
    
    if not user_query:
        return jsonify({"error": "Please provide a user query."}), 400
    
    responses = get_responses(user_query, top_n=5)
    return jsonify({"responses": responses})

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode for easy debugging
