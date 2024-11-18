import asyncio, os, time, uuid
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline, AutoTokenizer


application = Flask(__name__, template_folder='template')

executor = ThreadPoolExecutor(2)  # Create a ThreadPoolExecutor for handling multiple requests

# Load the Whisper ASR model and tokenizer
model = whisper.load_model("medium")
tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)

# Load the summarization model
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

@application.route("/")
def home():
    return render_template('mobile.html')

@application.route('/record', methods=['POST'])
def record_audio():
    action = request.form['action']

    if action == 'stop':
        audio_data = request.files['audio_data']
        
        unique_filename = f"{int(time.time())}_{uuid.uuid4()}.wav"
        
        audio_data.save(unique_filename)
        # Use ThreadPoolExecutor to run the processing function asynchronously
        future = executor.submit(transcribe_and_summarize, unique_filename)
        
        print("Audio saved as: ", unique_filename)
        
        # Return a response immediately without waiting for the processing to complete
        transcription, bullet_points = future.result()  # Get the results from the future
        return jsonify(transcription=transcription, bullet_points=bullet_points)

def transcribe_and_summarize(audio_path):
    options = {"fp16": False, "language": "English", "task": "transcribe"}
    results = model.transcribe(audio_path, **options)
    transcription_text = results["text"]

    list_text = summarizer(transcription_text, max_length=55, min_length=20, do_sample=False)
    summary = list_text[0].get('summary_text')
    bullet_points = summary

    print(transcription_text)
    print()
    print(bullet_points)
    return transcription_text, bullet_points

if __name__ == '__main__':
    application.run()

# application = Flask(__name__)
