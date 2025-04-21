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
from asr_utils import transcribe_audio

# Import moviepy
try:
    import moviepy
    from moviepy import AudioFileClip  # Updated import
except ImportError as e:
    print(f"Failed to import moviepy: {str(e)}")
    raise

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

# Load trained classification model
logger.info("Loading best_model.pkl...")
clf = joblib.load("best_model.pkl")

app.config['DB_HOST'] = '127.0.0.1'
app.config['DB_USER'] = 'root'        # <- replace with your MySQL username
app.config['DB_PASSWORD'] = 'admin123'    # <- replace with your MySQL password
app.config['DB_NAME'] = 'Alzwhisper'

from functools import wraps
from functools import wraps

def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        if not token:
            return jsonify({'error': 'Token missing'}), 401

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            user_id = payload['user_id']

            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM sessions WHERE user_id = %s AND jwt_token = %s", (user_id, token))
            session = cursor.fetchone()

            if not session:
                return jsonify({'error': 'Invalid or expired session'}), 403

            request.user = payload  # Optional: Attach to request

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return wrapper


# API route to handle Dashboard
@app.route('/api/dashboard', methods=['GET'])
@token_required
def dashboard():
    user = request.user  # Get user info attached to request by token_required
    return jsonify({
        'user': {
            'id': user['user_id'],
            'name': user.get('name', 'Unknown User'),
        },
        'message': 'Welcome to your dashboard!',
    })
# Database Connection Function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',#app.config['DB_HOST'],
            user='root' ,#app.config['DB_USER'],
            password='1234' ,#app.config['DB_PASSWORD'],
            database='Alzwhisper'#app.config['DB_NAME']
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Database connection error: {e}")
    return None

# Signup API Route
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email,  password]):
        return jsonify({"error": "All fields are required"}), 400

    db = get_db_connection()
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = db.cursor()
        query = """
            INSERT INTO users (name, email, password)
            VALUES (%s, %s,  %s)
        """
        cursor.execute(query, (name, email, password))
        db.commit()
        cursor.close()
        return jsonify({"message": "User created successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500
    finally:
        db.close()



@app.route('/api/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # Validate user
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    
    if user:
        # Generate JWT
        payload = {
            'user_id': user['user_id'],
            'email': user['email'],
            'exp': datetime.utcnow() + timedelta(hours=12)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

        # Store session
        cursor.execute("INSERT INTO sessions (user_id, jwt_token) VALUES (%s, %s)", (user['user_id'], token))
        db.commit()

        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'name': user['name'],
                'email': user['email'],
                'user_id': user['user_id']
            }
        }), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/logout', methods=['POST'])
@token_required
def logout():
    token = request.headers.get('Authorization').replace('Bearer ', '')
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM sessions WHERE jwt_token = %s", (token,))
    db.commit()
    return jsonify({'message': 'Logged out successfully'})


@app.route('/api/test_history', methods=['POST'])
# @token_required  # Ensure the user is logged in via the token
def test_history():
    # Extract data from the request body
    data = request.get_json()

    # Check if user_id is in the request
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Retrieve the user information from the JWT token (if necessary)
    # try:
    #     user = get_jwt_identity()  # Get the user information from the token (if needed)
    # except Exception as e:
    #     logger.error(f"Error extracting user from token: {e}")
    #     return jsonify({"error": "Unauthorized"}), 401
    
    db = get_db_connection()
    if db:
        try:
            cursor = db.cursor(dictionary=True)
            cursor.execute("""
                SELECT r.prediction, r.confidence, r.date, u.name
                FROM records r
                JOIN users u ON r.user_id = u.user_id
                WHERE r.user_id = %s
                ORDER BY r.date DESC
            """, (user_id,))
            records = cursor.fetchall()

            if not records:
                return jsonify({"message": "No test history found"}), 404

            return jsonify({"history": records}), 200
        except Exception as e:
            logger.error(f"Error fetching test history: {e}")
            return jsonify({"error": "Failed to fetch test history"}), 500
        finally:
            db.close()

    return jsonify({"error": "Database connection failed"}), 500

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        logger.error("No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['file']
    filename = file.filename
    logger.info(f"Received file: {filename}")
    filepath = os.path.join("Uploads", filename)
    file.save(filepath)
    user_id = request.form.get('user_id')

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
    transcript = transcribe_audio(filepath)

    print(f"📄 Transcript: {transcript}")

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

    # Predict using the model
    logger.info("Making prediction...")
    prediction = clf.predict(df_selected)[0]
    confidence = clf.predict_proba(df_selected).max()

    logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
    db = get_db_connection()
    if db:
        try:
            cursor = db.cursor()
            insert_query = """
                INSERT INTO records (user_id, confidence, prediction, date)
                VALUES (%s, %s, %s, %s)
            """
            today = datetime.utcnow().date()
            cursor.execute(insert_query, (user_id, confidence, str(prediction), today))
            db.commit()
        except Exception as e:
            logger.error(f"Error saving record: {e}")
        finally:
            db.close()
    return jsonify({
        "prediction": str(prediction),
        "confidence": float(confidence),
        "message": "Audio analyzed successfully"
    })

if __name__ == '__main__':
    os.makedirs("Uploads", exist_ok=True)
    logger.info("Starting Flask server...")
    app.run(debug=True)