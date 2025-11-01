"""
SENTIMENT ANALYSIS APP USING MANUAL MODEL COMPONENTS
====================================================
This approach manually handles each component (tokenizer, model, post-processing)
providing more control and educational insight into the transformer workflow.
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import torch.nn.functional as F

# Load model and tokenizer separately for more control
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

def sentiment_analyser(text):
    """
    Process text through manual sentiment analysis pipeline
    
    Steps:
    1. Tokenization: Convert text to model inputs
    2. Model Inference: Get raw logits
    3. Softmax: Convert logits to probabilities
    4. Post-processing: Get final label and confidence
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Formatted result with label and confidence score
    """
    lang = detect(text)


    if lang != 'en':
      # the language is not english translation it in english
        result = translator(text)
        text = result[0]['translation_text']

    else:
        # text is English language, can stay without translation
        pass
    # Step 1: Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Step 2: Get model predictions (disable gradient for inference)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Step 3: Apply softmax to convert logits to probabilities
    probabilities = F.softmax(outputs.logits, dim=-1)
    
    # Step 4: Get the predicted class and confidence score
    confidence, predicted_class = torch.max(probabilities, dim=1)
    
    # Map class index to human-readable label
    labels = ["NEGATIVE", "POSITIVE"]
    predicted_label = labels[predicted_class.item()]
    
    return f"Label: {predicted_label}, Confidence: {confidence.item():.4f}"

# Create Gradio interface
demo = gr.Interface(
    fn=sentiment_analyser,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="Sentiment Analysis App (Manual Components)",
    description="Enter text to analyze its sentiment (Positive/Negative) using DistilBERT model with manual tokenizer and model.",
    examples=[
        ["I love this product! It's absolutely amazing!"],
        ["This is the worst experience I've ever had."],
        ["The movie was okay, nothing special."],
        ["I'm so happy with my purchase, it exceeded all expectations!"],
        ["Terrible service and poor quality product."],
        ["It's decent for the price, but could be better."],
        ["This is fantastic! I would highly recommend it to everyone!"],
        ["I'm very disappointed with this, it broke after one use."]
    ]
)

if __name__ == "__main__":
    demo.launch()