from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math

app = Flask(__name__)

print("Loading models...")

# Classifier model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased"
)

# Language model for perplexity
lm_name = "gpt2"
lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_name)

print("Models loaded successfully.")


def calculate_perplexity(text):

    encodings = lm_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = lm_model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss

    perplexity = math.exp(loss.item())

    return round(perplexity, 2)


def analyze_text(text):

    # classifier prediction
    result = classifier(text)[0]
    score = round(result["score"] * 100, 2)

    # perplexity
    ppl = calculate_perplexity(text)

    if ppl < 40:
        verdict = "Likely AI Generated"
    else:
        verdict = "Likely Human Written"

    return score, ppl, verdict


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    text = data.get("text", "")

    if len(text.split()) < 10:
        return jsonify({
            "error": "Please enter at least 10 words"
        }), 400

    score, perplexity, verdict = analyze_text(text)

    return jsonify({
        "ai_score": score,
        "perplexity": perplexity,
        "verdict": verdict
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)