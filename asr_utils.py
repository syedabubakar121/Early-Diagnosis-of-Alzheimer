# asr_utils.py

import os, base64, json, hashlib, hmac, websocket, threading, time
from datetime import datetime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode
from pydub import AudioSegment
import torch
from transformers import BertTokenizer, BertModel

# Define folders
UPLOAD_FOLDER = "uploads"
PCM_FOLDER = "converted"
TRANSCRIPT_FOLDER = "transcript"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PCM_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
# --------- CONFIG (Add your actual credentials) ---------

APPID = 'ga637f40'
APIKey = 'e83d852f6c7ea7fead63ccaeee9271e9'
APISecret = '2b8d35fce36129e4b290158909e144c8'

# --------- ASR Client (Xunfei / iFLYTEK) ---------
class ASRClient:
    def __init__(self, app_id, api_key, api_secret):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    def create_url(self):
        host = "iat-api-sg.xf-yun.com"
        path = "/v2/iat"
        schema = "wss"
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        signature_sha = hmac.new(self.api_secret.encode(), signature_origin.encode(), hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode()
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(authorization_origin.encode()).decode()
        values = {"host": host, "date": date, "authorization": authorization}
        return f"{schema}://{host}{path}?" + urlencode(values)

    def transcribe_chunk(self, chunk_file):
        url = self.create_url()
        result = ""

        def on_message(ws, message):
            nonlocal result
            message = json.loads(message)
            if message["code"] == 0:
                words = message["data"]["result"]["ws"]
                for w in words:
                    for c in w["cw"]:
                        result += c["w"] + " "

        def on_close(ws, *args):
            print("🔴 Connection closed")

        def on_open(ws):
            def run():
                with open(chunk_file, "rb") as f:
                    while True:
                        buf = f.read(8000)
                        if not buf:
                            break
                        d = {
                            "common": {"app_id": self.app_id},
                            "business": {"language": "en_us", "domain": "iat", "vad_eos": 5000},
                            "data": {
                                "status": 0,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode(),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(d))
                        time.sleep(0.04)
                    ws.send(json.dumps({"data": {"status": 2}}))
            thread=threading.Thread(target=run).start()
            thread.start()
            thread.join()
        ws = websocket.WebSocketApp(url, on_message=on_message, on_close=on_close)
        ws.on_open = on_open
        ws.run_forever()
        return result.strip()

# --------- Audio Processing ---------
# def convert_to_pcm(input_audio, output_pcm):
#     audio = AudioSegment.from_file(input_audio)
#     audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
#     audio.export(output_pcm, format="s16le")
#     return output_pcm

# def split_audio(audio_file, chunk_length_ms=60000):
#     audio = AudioSegment.from_file(audio_file, format="raw", frame_rate=16000, channels=1, sample_width=2)
#     return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# def transcribe(pcm_file, client):
#     chunks = split_audio(pcm_file)
#     full_transcript = ""
#     for i, chunk in enumerate(chunks):
#         chunk_path = os.path.join(PCM_FOLDER, f"chunk_{i}.pcm")
#         chunk.export(chunk_path, format="s16le")
#         print(f"🧠 Transcribing chunk {i+1}/{len(chunks)}")
#         transcript = client.transcribe_chunk(chunk_path)
#         full_transcript += transcript + " "
#     return full_transcript.strip()

# def transcribe_audio(input_audio_path):
#     pcm_path = convert_to_pcm(input_audio_path, os.path.join(PCM_FOLDER, "temp.pcm"))
#     client = ASRClient(APPID, APIKey, APISecret)
#     return transcribe(pcm_path, client)


# def convert_to_pcm(input_audio, output_pcm):
#     audio = AudioSegment.from_file(input_audio)
#     audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
#     audio.export(output_pcm, format="s16le")
#     return output_pcm

# def transcribe_audio(input_audio_path):
#     pcm_path = convert_to_pcm(input_audio_path, os.path.join(PCM_FOLDER, "temp.pcm"))
#     client = ASRClient(APPID, APIKey, APISecret)
#     print("🧠 Transcribing full audio")
#     transcript = client.transcribe_chunk(pcm_path)  # Transcribe the whole audio file directly
#     return transcript


# Convert input audio to PCM format
def convert_to_pcm(input_audio, output_pcm):
    audio = AudioSegment.from_file(input_audio)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_pcm, format="s16le")
    return output_pcm

# Split the PCM audio into chunks
def split_audio(audio, chunk_length_ms=60000):
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Transcribe the chunks and return full transcript
def transcribe(pcm_file, client):
    audio = AudioSegment.from_file(pcm_file, format="raw", frame_rate=16000, channels=1, sample_width=2)
    chunks = split_audio(audio)
    full_transcript = ""
    
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(PCM_FOLDER, f"chunk_{i}.pcm")
        chunk.export(chunk_path, format="s16le")
        print(f"🧠 Transcribing chunk {i+1}/{len(chunks)}")
        transcript = client.transcribe_chunk(chunk_path)
        full_transcript += transcript + " "
    
    return full_transcript.strip()

# Main function to process audio and get transcript
def transcribe_audio(input_audio_path):
    pcm_path = convert_to_pcm(input_audio_path, os.path.join(PCM_FOLDER, "temp.pcm"))
    client = ASRClient(APPID, APIKey, APISecret)  # You will need to define your client and keys
    full_transcript = transcribe(pcm_path, client)
    
    # Save the full transcript to a file
    transcript_path = os.path.join(TRANSCRIPT_FOLDER, "full_transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(full_transcript)
    print(f"✅ Full transcript saved at: {transcript_path}")
    
    return full_transcript.strip()