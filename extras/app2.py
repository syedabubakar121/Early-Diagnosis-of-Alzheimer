# # # from flask import Flask, request, jsonify
# # # from asr_utils import transcribe_audio
# # # from transformers import BertTokenizer, BertForSequenceClassification
# # # import torch
# # # from sklearn.decomposition import PCA
# # # import numpy as np
# # # import os

# # # app = Flask(__name__)

# # # # Device config
# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # Load tokenizer and fine-tuned model
# # # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Change this path if using a fine-tuned local model
# # # bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # # # Load the fine-tuned model weights
# # # bert_model.load_state_dict(torch.load("v3-fine_tuned_bert.pth", map_location=device))
# # # bert_model.eval()
# # # print("✅ Fine-tuned model loaded successfully.")

# # # # PCA config
# # # pca = PCA(n_components=50)

# # # # Feature extraction function
# # # def extract_bert_features(text):
# # #     # Tokenize and prepare input for BERT
# # #     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
# # #     inputs = {key: val.to(device) for key, val in inputs.items()}

# # #     # Get the embeddings from the BERT model
# # #     with torch.no_grad():
# # #         outputs = bert_model.bert(**inputs)
# # #         cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token (first token)
    
# # #     # Convert the embedding to a numpy array and reduce its dimensionality using PCA
# # #     cls_vector = cls_embedding.squeeze().cpu().numpy().reshape(1, -1)
# # #     reduced_vector = pca.fit_transform(cls_vector)  # Fit + transform for now (you might want to refit on training data)
    
# # #     return {f"ling_{i+1}": float(val) for i, val in enumerate(reduced_vector[0])}

# # # # Allowed audio file extensions
# # # allowed_extensions = {'wav', 'mp3', 'flac'}

# # # def allowed_file(filename):
# # #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# # # # Upload endpoint
# # # @app.route('/upload', methods=['POST'])
# # # def upload_audio():
# # #     if 'audio' not in request.files:
# # #         return jsonify({"error": "No audio file provided"}), 400

# # #     audio_file = request.files['audio']

# # #     if not allowed_file(audio_file.filename):
# # #         return jsonify({"error": "Invalid audio file format"}), 400

# # #     save_path = os.path.join("uploads", audio_file.filename)
# # #     os.makedirs("uploads", exist_ok=True)
# # #     audio_file.save(save_path)

# # #     print(f"🎙 Audio saved at: {save_path}")

# # #     # Transcribe the audio file
# # #     transcript = transcribe_audio(save_path)
# # #     print(f"📄 Transcript: {transcript}")

# # #     # Extract BERT features
# # #     features = extract_bert_features(transcript)

# # #     return jsonify({
# # #         "transcript": transcript,
# # #         "linguistic_features": features
# # #     })

# # # if __name__ == '__main__':
# # #     app.run(debug=True)

# from flask import Flask, request, jsonify
# import torch
# import os
# import joblib
# import numpy as np
# from transformers import BertTokenizer, BertModel, BertForSequenceClassification
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from asr_utils import transcribe_audio  # Assuming you have this helper function for transcription

# # Load models and objects
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Or use fine_tuned_bert_askari
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # # # Load the fine-tuned model weights
# bert_model.load_state_dict(torch.load("fine_tuned_bert_askari.pth", map_location=device))
# # # bert_model.eval()
# bert_model.eval()

# # Load pre-trained models and scaler
# tfidf = joblib.load("tuned_tfidf_vectorizer.pkl")
# scaler = joblib.load("tuned_scaler.pkl")
# pca = joblib.load("tuned_pca.pkl")

# # BERT feature extraction (CLS token)
# def extract_bert_cls(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#         cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
#     return cls_embedding

# # Full pipeline for extracting features
# def extract_50_features(text):
#     # Step 1: Get TF-IDF (301) and BERT CLS (768)
#     tfidf_vector = tfidf.transform([text]).toarray().squeeze()
#     bert_vector = extract_bert_cls(text)
    
#     # Step 2: Combine features
#     combined = np.concatenate([bert_vector, tfidf_vector]).reshape(1, -1)

#     # Step 3: Apply Scaler
#     scaled = scaler.transform(combined)

#     # Step 4: Apply PCA to get 50 features
#     pca_features = pca.transform(scaled)

#     return pca_features.squeeze()  # shape: (50,)

# # Flask API setup
# app = Flask(__name__)

# # Endpoint to upload audio and extract features
# @app.route('/upload', methods=['POST'])
# def upload_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file provided"}), 400

#     audio_file = request.files['audio']
    
#     if not allowed_file(audio_file.filename):
#         return jsonify({"error": "Invalid audio file format"}), 400

#     save_path = os.path.join("uploads", audio_file.filename)
#     os.makedirs("uploads", exist_ok=True)
#     audio_file.save(save_path)
#     print(f"🎙 Audio saved at: {save_path}")

#     # Transcribe audio
#     transcript = transcribe_audio(save_path)
#     print(f"📄 Transcript: {transcript}")

#     # Extract 50 features
#     features = extract_50_features(transcript)

#     return jsonify({
#         "transcript": transcript,
#         "linguistic_features": features.tolist()  # Convert numpy array to list for JSON response
#     })

# # Helper function to check if file is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'flac'}

# # Run Flask app
# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# import re
# import torch
# import numpy as np
# from flask import Flask, request, jsonify
# from transformers import BertTokenizer, BertForSequenceClassification
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import pickle
# import joblib
# from asr_utils import transcribe_audio 

# # Initialize Flask app
# app = Flask(__name__)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load fine-tuned BERT model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model_with_head = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model_with_head.load_state_dict(torch.load("fine_tuned_bert_askari.pth", map_location=device))
# model_with_head.to(device)
# model_with_head.eval()
# bert_encoder = model_with_head.bert


# tfidf = joblib.load("tuned_tfidf_vectorizer.pkl")
# scaler = joblib.load("tuned_scaler.pkl")
# pca = joblib.load("tuned_pca.pkl")

# # Extract BERT CLS embedding from encoder
# def extract_bert_features(text):
#     tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     tokens = {key: val.to(device) for key, val in tokens.items()}
#     with torch.no_grad():
#         outputs = bert_encoder(**tokens)
#     cls_embedding = outputs.logits[:, 0]  # Get [CLS]-like representation from logits
#     return cls_embedding.cpu().numpy().flatten()
   

# # Feature extraction pipeline
# def extract_50_features(text):
#     bert_vector = np.array(extract_bert_features(text) )
#     tfidf_vector = TfidfVectorizer(max_features=300)
#     combined_features = np.hstack([bert_vector, tfidf_vector])
#     scaled = scaler.fit_transform([combined_features])
#     reduced = pca.fit_transform(scaled).squeeze()
#     return reduced

# # Flask endpoint
# @app.route("/upload", methods=["POST"])
# def upload_audio():
#     try:
#         if 'audio' not in request.files:
#             return jsonify({"error": "No audio file provided"}), 400

#         audio_file = request.files['audio']
        
        

#         save_path = os.path.join("uploads", audio_file.filename)
#         os.makedirs("uploads", exist_ok=True)
#         audio_file.save(save_path)
#         print(f"🎙 Audio saved at: {save_path}")

#         # Transcribe audio
#         # transcript = transcribe_audio(save_path)
#         transcript="Tell  me  everything  that's  going  on . And  all  of  a  sudden  somebody  stepped  it  up .
#                     Turn  over  a  dish  ,  all  over  the  floor .  Accept  it .  It  did  not  try  to  cook  it  didn't ,  um  splash  from  the  splashed  from  the  sink  but  not  for  me .  No .  To  to  try  to  get  too  much  out  of  them .  That's  okay .  Is  there  anything  
#                     else                ?  The  kids  are  going  to  get  to  crack  on  the  head .  Uh  . You  know  it's  it's  so  sometimes  I  see  it  very  clearly .  And  uh ,  at  times  I  see  I  have  a  weak  image  to  speak .  And  sometimes  I I  just  don't ,  you  know ,  is  there  anything  else ? Is  there  anything  else  in  '
#                     'tha                t  picture  you  wanna  tell  me  or  do  you  think  you've  told  me  everything ?  Ok ,  thank  you .  All  right"
        
#         print(f"📄 Transcript: {transcript}")

#         # Extract transcript from request
#         # transcript = request.form.get("transcript")
#         if not transcript:
#             return jsonify({"error": "Transcript not provided"}), 400

#         # Feature extraction
#         features = extract_50_features(transcript)
#         feature_dict = {f"ling_{i+1}": float(val) for i, val in enumerate(features)}

#         return jsonify({
#             "status": "success",
#             "features": feature_dict,
#             "transcript":transcript
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Run server
# if __name__ == "__main__":
#     app.run(debug=True)

import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from asr_utils import transcribe_audio  # If you're using ASR, otherwise comment this

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load fine-tuned BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_with_head = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_with_head.load_state_dict(torch.load("fine_tuned_bert_askari.pth", map_location=device))
model_with_head.to(device)
model_with_head.eval()
bert_encoder = model_with_head.bert

# Load pre-fitted TF-IDF, scaler, and PCA
tfidf = joblib.load("tuned_tfidf_vectorizer.pkl")
scaler = joblib.load("tuned_scaler.pkl")
pca = joblib.load("tuned_pca.pkl")

# Extract BERT CLS embedding
def extract_bert_features(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        outputs = bert_encoder(**tokens)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Extracting the CLS token embedding
    return cls_embedding

# Combine BERT + TF-IDF + scale + PCA
def extract_301_features(text):
    bert_vector = extract_bert_features(text)
    tfidf_vector = tfidf.transform([text]).toarray().squeeze()

    # Limit BERT features to 1 feature (CLS token) and combine with TF-IDF
    if len(bert_vector) > 1:
        bert_vector = bert_vector[:1]  # Only keep 1 feature (CLS token)

    combined_features = np.hstack([bert_vector, tfidf_vector])  # Combine BERT (1 feature) + TF-IDF (300 features)

    # Ensure total features before PCA is 301
    if combined_features.shape[0] != 301:
        raise ValueError(f"Feature size mismatch: Expected 301 features, but got {combined_features.shape[0]} features.")
    
    # Scaling
    scaled = scaler.transform([combined_features])

    # PCA for dimensionality reduction (e.g., reducing to 50 features)
    reduced = pca.transform(scaled).squeeze()
    return reduced

# Flask endpoint
@app.route("/upload", methods=["POST"])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", audio_file.filename)
        audio_file.save(save_path)
        print(f"🎙 Audio saved at: {save_path}")

        # Transcribe audio
        # transcript = transcribe_audio(save_path)
        transcript = transcribe_audio(save_path)

        print(f"📄 Transcript: {transcript}")

        if not transcript:
            return jsonify({"error": "Transcript not provided"}), 400

        features = extract_301_features(transcript)  # Now this will return 301 features
        feature_dict = {f"ling_{i+1}": float(val) for i, val in enumerate(features)}

        return jsonify({
            "status": "success",
            "features": feature_dict,
            "transcript": transcript
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True)
