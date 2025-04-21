

import sys
import os

# Prioritize virtual environment
venv_site_packages = r'S:\FYP\Development\venv\Lib\site-packages'
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)
else:
    sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)

from flask import Flask, request, jsonify,session
from flask_cors import CORS
import torch
import torchaudio
import pandas as pd
import json
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import tempfile
import logging
import soundfile as sf
import librosa
from auth_routes import auth_bp
import mysql.connector
from mysql.connector import Error
import jwt
import datetime
import secrets
from datetime import datetime, timedelta


# Import moviepy
try:
    import moviepy
    from moviepy import AudioFileClip  # Updated import
except ImportError as e:
    print(f"Failed to import moviepy: {str(e)}")
    raise
import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from asr_utils import transcribe_audio 

app = Flask(__name__)
# CORS(app)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
SECRET_KEY = secrets.token_hex(32)  # Or load from env

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Moviepy version: {moviepy.__version__}")
logger.info(f"Moviepy path: {moviepy.__file__}")

# Load Wav2Vec2 model and processor
logger.info("Loading Wav2Vec2 model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Load selected features list
logger.info("Loading features.json...")
with open("features.json") as f:
    selected_features = json.load(f)["selected_features"]


# Initialize Flask app

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

        df_ling = pd.DataFrame([feature_dict])


        
        
        filename= request.files['audio']
        # Convert .webm to .wav if necessary
        temp_wav = None
        # if filename.endswith('.webm'):
        #     logger.info("Converting .webm to .wav...")
        #     try:
        #         temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        #         audio_clip = AudioFileClip(filepath)
        #         audio_clip.write_audiofile(temp_wav, codec='pcm_s16le')
        #         audio_clip.close()
        #         logger.info(f"Converted to: {temp_wav}")
        #         filepath = temp_wav
        #     except Exception as e:
        #         logger.error(f"Conversion failed: {str(e)}")
        #         return jsonify({"error": f"Failed to convert .webm to .wav: {str(e)}"}), 500
        if filename.endswith('.webm'):
            logger.info("Converting .webm to .wav using librosa and soundfile...")
            try:
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                y, sr = librosa.load(filepath, sr=16000)  # Always convert to 16kHz mono
                sf.write(temp_wav, y, sr)
                logger.info(f"Converted to: {temp_wav}")
                filepath = temp_wav
            except Exception as e:
                logger.error(f"Conversion failed: {str(e)}")
                return jsonify({"error": f"Failed to convert .webm to .wav: {str(e)}"}), 500
            
        # Load and resample audio
        try:
            logger.info(f"Loading audio: {filepath}")
            waveform, sample_rate = torchaudio.load(filepath)
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate} to 16000 Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
        except Exception as e:
            logger.error(f"Audio loading failed: {str(e)}")
            return jsonify({"error": f"Failed to load audio: {str(e)}"}), 500
        finally:
            # Clean up files
            if os.path.exists(filepath) and filepath != temp_wav:
                logger.info(f"Removing original file: {filepath}")
                os.remove(filepath)
            if temp_wav and os.path.exists(temp_wav):
                logger.info(f"Removing temp file: {temp_wav}")
                os.remove(temp_wav)

        # Extract features using Wav2Vec2
        logger.info("Extracting features with Wav2Vec2...")
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values).last_hidden_state
        features = outputs.mean(dim=1).numpy().flatten()

        # Convert to dict and DataFrame
        all_features_dict = {f"Feature_{i}": float(val) for i, val in enumerate(features)}
        df_all = pd.DataFrame([all_features_dict])

        # Save full CSV for debugging (optional)
        base_name = filename.rsplit('.', 1)[0]
        csv_path = os.path.join("Uploads", f"all_features_{base_name}.csv")
        df_all.to_csv(csv_path, index=False)
        logger.info(f"Saved features to: {csv_path}")

        # Select only required features
        df_selected = df_all[selected_features]

        combined_features = np.concatenate((df_selected,  df_ling))
        fused_model = joblib.load("l3-fused-voting.pkl")
        prediction = fused_model.predict([combined_features])[0]
        probability = fused_model.predict_proba([combined_features]).max()

        return jsonify({
            "prediction": int(prediction),
            "confidence": float(probability),
            "message": "Prediction successful"
        })

    except Exception as e:
        logger.error(f"❌ Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500



# Run server
if __name__ == "__main__":
    app.run(debug=True)
