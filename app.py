from flask import Flask, request, render_template
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Let's load knowledge base json file
with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

# Let's load the TensorFlow model and tokenizer
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/get_drug", methods=["POST"])
def get_drug():
    user_input = request.form["symptoms"]

    # Let's define the possible conditions from the knowledge base
    possible_conditions = list(knowledge_base.keys())

    # Let's tokenize the input
    inputs = tokenizer(
        [user_input] * len(possible_conditions), 
        possible_conditions, 
        return_tensors="tf", 
        truncation=True, 
        padding=True
    )

    # Let's perform the inference
    outputs = model(inputs)
    logits = outputs.logits
    scores = tf.nn.softmax(logits, axis=1)

    entailment_scores = scores[:, 2].numpy()

    # Let's set a confidence threshold
    threshold = 0.8
    predicted_conditions = [
        possible_conditions[i] for i, score in enumerate(entailment_scores) if score > threshold
    ]

    # Let's retrieve drug information for each predicted condition
    drug_suggestions = []
    for condition in predicted_conditions:
        if condition in knowledge_base:
            drug_suggestions.append({
                "condition": condition,
                "drug": knowledge_base[condition]["drug"]
            })

    # Let's render the results in an HTML template
    if drug_suggestions:
        return render_template("results.html", suggestions=drug_suggestions, symptoms=user_input)
    else:
        return render_template("results.html", error="No matching conditions found for the given symptoms", symptoms=user_input)

if __name__ == "__main__":
    app.run(debug=True)
