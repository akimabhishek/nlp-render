import pandas as pd
import difflib
import os
import requests
import gensim
from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec

app = FastAPI()

MODEL_PATH = "friends_word2vec_with_phrases.model"
MODEL_URL = "https://huggingface.co/akimabhi/friends-word2vec/resolve/main/friends_word2vec_with_phrases.model"

# Step 1: Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded!")

# Step 2: Load the model
print("📦 Loading Word2Vec model...")
model = gensim.models.Word2Vec.load(MODEL_PATH)
print("✅ Model loaded!")

@app.get("/")
def root():
    return {"message": "Friends Word2Vec API"}

@app.get("/similar")
def similar(word: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(word.lower(), topn=topn)
        return {"word": word, "similar": similar_words}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{word}' not in vocabulary")

@app.get("/similarity")
def similarity(word1: str, word2: str):
    try:
        sim = model.wv.similarity(word1.lower(), word2.lower()) * 100
        return {"word1": word1, "word2": word2, "similarity": sim}
    except KeyError:
        raise HTTPException(status_code=404, detail="One or both words not in vocabulary")

@app.get("/traits")
def traits(name: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(name.lower(), topn=topn)
        traits = [word for word, _ in similar_words]
        return {"character": name, "traits": traits}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{name}' not in vocabulary")

@app.get("/match")
def match(name: str, topn: int = 3):
    try:
        matches = model.wv.most_similar(name.lower(), topn=topn)
        return {"character": name, "most_similar_characters": matches}
    except KeyError:
        raise HTTPException(status_code=404, detail="Character not in vocabulary")

@app.get("/analogy")
def analogy(pos1: str, neg: str, pos2: str):
    try:
        result = model.wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=1)
        return {"result": result}
    except KeyError:
        raise HTTPException(status_code=404, detail="One or more words not found")

@app.get("/help")
def help():
    return {
        "message": "Welcome to the Friends Word2Vec API. Here are the available endpoints:",
        "endpoints": {
            "/": "Health check. Returns a welcome message.",
            "/similar?word=WORD&topn=N": "Returns top N similar words to the given WORD.",
            "/similarity?word1=WORD1&word2=WORD2": "Returns similarity score between WORD1 and WORD2.",
            "/traits?name=NAME&topn=N": "Returns N traits (similar words) for a character NAME.",
            "/analogy?positive=word1,word2&negative=word3": "Performs word analogy using positive and negative words.",
            "/match?positive=WORD&candidates=word1,word2": "Finds best match for WORD from a list of candidate words.",
            "/help": "You're here! Lists all available API endpoints."
        },
        "note": "Use lowercase words. Words not in vocabulary will return 404."
    }

print("✅ App started on Railway!")
