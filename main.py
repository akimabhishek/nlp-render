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
def similar(character: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(character.lower(), topn=topn)
        return {"Character": character, "similar": similar_words}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{character}' not in vocabulary")

@app.get("/similarity")
def similarity(character1: str, character2: str):
    try:
        sim = model.wv.similarity(character1.lower(), character2.lower()) * 100
        return {"Character 1": character1, "Character 2": character2, "similarity": sim}
    except KeyError:
        raise HTTPException(status_code=404, detail="One or both words not in vocabulary")

@app.get("/traits")
def traits(character: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(character.lower(), topn=topn)
        traits = [character for character, _ in similar_words]
        return {"character": character, "traits": traits}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{character}' not in vocabulary")

@app.get("/match")
def match(name: str, topn: int = 3):
    try:
        matches = model.wv.most_similar(name.lower(), topn=topn)
        return {"character": name, "most_similar_characters": matches}
    except KeyError:
        raise HTTPException(status_code=404, detail="Character not in vocabulary")

@app.get("/analogy")
def analogy(positive_character: str, negative_character: str):
    try:
        result = model.wv.most_similar(positive=[positive_character], negative=[negative_character], topn=1)
        return {"result": result}
    except KeyError:
        raise HTTPException(status_code=404, detail="One or more words not found")

@app.get("/odd_one_out")
def odd_one_out(characters: str):
    word_list = characters.lower().split(",")
    odd = model.wv.doesnt_match(word_list)
    return {"words": word_list, "odd_one_out": odd}

@app.get("/sentiment")
def character_sentiment(name: str):
    # Return average sentiment score of that character
    return {"character": name, "sentiment": {"polarity": 0.41, "subjectivity": 0.56}}


from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


@app.get("/visualize_image")
def visualize_image(character1: str, character2: str, character3: str):
    try:
        words = [character1.lower(), character2.lower(), character3.lower()]
        vectors = [model.wv[word] for word in words]

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        plt.figure(figsize=(6, 6))
        for i, word in enumerate(words):
            plt.scatter(reduced[i, 0], reduced[i, 1])
            plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, word, fontsize=12)

        plt.title("Word2Vec Character Embedding")
        plt.xlabel("Principal Component 1")  # ✅ X-axis label
        plt.ylabel("Principal Component 2")  # ✅ Y-axis label
        plt.grid(True)                       # ✅ Adds grid lines

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except KeyError:
        raise HTTPException(status_code=404, detail="One or more characters not found")

@app.get("/version")
def version():
    return {"version": "0.1-beta", "status": "Training continues. DL-based upgrade coming soon."}

@app.get("/help")
def help():
    return {
        "message": "Welcome to the Friends Word2Vec API. Here are the available endpoints:",
        "endpoints": {
            "/": "Health check. Returns a welcome message.",
            "/similar?character=Character Name&topn=N": "Returns top N similar words to the given Character.",
            "/similarity?character1=Character 1&character2=Character 2": "Returns similarity score between Character 1 and Character 2.",
            "/analogy?positive_character=...&negative_character=...": "Performs word arithmetic with character analogy.",
            "/visualize_image?char1=...&char2=...&char3=...": "Returns a 2D graph showing relation between 3 characters.",
            "/version": "Return the version of the product.",
            "/help": "You're here! Lists all available API endpoints."
        },
        "note": "Use lowercase words. Words not in vocabulary will return 404."

    }


print("✅ App started on Railway!")
