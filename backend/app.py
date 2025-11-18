from flask import Flask, request, jsonify
from deepface import DeepFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import urllib.request
import csv
import base64
import os

app = Flask(__name__)

def get_text_emotion(text):
    MODEL = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    text = " ".join(new_text)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = softmax(outputs.logits[0].detach().numpy())
    url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
    with urllib.request.urlopen(url) as f:
        labels = [row[1] for row in csv.reader(f.read().decode('utf-8').split("\n"), delimiter='\t') if len(row) > 1]
    return labels[scores.argmax()]

def get_image_emotion(img_path):
    try:
        result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "neutral"

MOOD_PLAYLISTS = {
    "joy": "37i9dQZF1DXdPec7aLTmlC",
    "sadness": "37i9dQZF1DXdFxB6aud8gK",
    "anger": "37i9dQZF1DX1s9knjP51Oa",
    "fear": "37i9dQZF1DX4H7mY2uF8QG",
    "neutral": "37i9dQZF1DX0XUsuxWHRQd",
    "surprise": "37i9dQZF1DX4JAvHpjipBk",
    "disgust": "37i9dQZF1DX0kbJZpiYdZl",
    "optimism": "37i9dQZF1DXdPec7aLTmlC"
}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    mood = "neutral"
    if data.get('type') == 'text':
        mood = get_text_emotion(data['text'])
    elif data.get('type') == 'image':
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        os.makedirs('temp', exist_ok=True)
        path = 'temp/upload.jpg'
        with open(path, 'wb') as f:
            f.write(img_bytes)
        mood = get_image_emotion(path)
        os.remove(path)
    playlist_url = f"https://open.spotify.com/playlist/{MOOD_PLAYLISTS.get(mood, MOOD_PLAYLISTS['neutral'])}"
    return jsonify({"emotion": mood, "playlist_url": playlist_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
