"""
SENTIMENT ANALYSIS APP USING HUGGING FACE PIPELINE
==================================================
This approach uses the simplified pipeline API from Hugging Face
which handles tokenization, model inference, and post-processing automatically.
"""

import gradio as gr
from transformers import pipeline

# Load the model using Hugging Face pipeline
# The pipeline automatically handles:
# - Tokenization
# - Model inference
# - Output processing
classifier = pipeline("text-classification", 
                     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analyser(text):
    """
    Process text through the sentiment analysis pipeline
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Formatted result with label and confidence score
    """
    # The pipeline returns a list of dictionaries
    result = classifier(text)[0]
    
    # Format the output
    return f"Label: {result['label']}, Confidence: {result['score']:.4f}"

# Create Gradio interface
demo = gr.Interface(
    fn=sentiment_analyser,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="Sentiment Analysis App (Auto Pipeline)",
    description="Enter text to analyze its sentiment (Positive/Negative) using DistilBERT model with pipeline.",
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