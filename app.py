
import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os
import uuid

# Load the model and tokenizer
model = tf.keras.models.load_model("sentiment_model.keras", compile=False)
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 200

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return f"{sentiment} ({prediction:.2f})"

def analyze_csv(file, chart_type):
    df = pd.read_csv(file.name)
    if "review" not in df.columns:
        return "CSV must contain a 'review' column.", None

    sequences = tokenizer.texts_to_sequences(df["review"].astype(str))
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    preds = model.predict(padded).flatten()
    sentiments = ["Positive" if p >= 0.5 else "Negative" for p in preds]
    df["Sentiment"] = sentiments

    # Visualization
    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    if chart_type == "Bar":
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=["lightgreen", "salmon"])
        ax.set_title("Sentiment Distribution - Bar Chart")
        ax.set_ylabel("Count")
    else:
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        ax.set_title("Sentiment Distribution - Pie Chart")

    chart_name = f"{uuid.uuid4()}.png"
    fig.savefig(chart_name)
    plt.close(fig)

    return df[["review", "Sentiment"]], chart_name

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Sentiment Analysis with LSTM")
    
    with gr.Tab("Single Review"):
        review_input = gr.Textbox(label="Enter your review", placeholder="Type a review here...")
        output = gr.Textbox(label="Sentiment")
        btn = gr.Button("Analyze")
        btn.click(predict_sentiment, inputs=review_input, outputs=output)

    with gr.Tab("CSV Upload"):
        file_input = gr.File(file_types=[".csv"], label="Upload CSV")
        chart_selector = gr.Radio(["Pie", "Bar"], value="Pie", label="Select Chart Type")
        csv_output = gr.Dataframe(label="Prediction Results")
        chart_output = gr.Image(label="Chart")
        submit = gr.Button("Analyze CSV")
        submit.click(analyze_csv, inputs=[file_input, chart_selector], outputs=[csv_output, chart_output])

demo.launch()
