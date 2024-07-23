# # import torch
# # # import os
# # # import textwrap

# # # import numpy as np
# # # import pandas as pd
# # # import seaborn as sns
# # # import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import Dataset, DataLoader

# # # from sklearn.model_selection import train_test_split
# # # from datasets import load_dataset, load_metric
# from transformers import T5Tokenizer, AdamW, T5ForConditionalGeneration

# # # from tqdm import tqdm
# # # from termcolor import colored
# # # from collections import defaultdict
# # # from IPython.display import display, HTML

# # # import nltk
# # # nltk.download('punkt')

# # # import warnings
# # # warnings.filterwarnings('ignore')
# # from torch.utils.data import Dataset, DataLoader

# class CNNDailyMailModel(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.model = Config.MODEL

#   def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
#     output = self.model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         decoder_attention_mask=decoder_attention_mask,
#         labels=labels,
#     )
#     return output.loss, output.logits
  

# class Config:
#   EPOCHS = 4
#   MAX_LEN = 512
#   HIG_MAX_LEN = 64
#   ART_MAX_LEN = 512
#   LEARNING_RATE = 3e-5
#   TRAIN_BATCH_SIZE = 4
#   VALID_BATCH_SIZE = 4
#   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   CHECKPOINT = "t5-base"
#   MODEL_PATH = "t5ass.bin"

#   TOKENIZER = T5Tokenizer.from_pretrained(CHECKPOINT)
#   MODEL = T5ForConditionalGeneration.from_pretrained(CHECKPOINT, return_dict=True)

# class Process:
#   def __init__(self, article_text):
#     self.article_text = article_text
#     self.art_max_len = Config.ART_MAX_LEN
#     self.tokenizer = Config.TOKENIZER


#   def pre_process(self):
#     article_text = str(self.article_text)
#     article_text = " ".join(article_text.split())

#     inputs = self.tokenizer.batch_encode_plus(
#         [article_text],
#         max_length=self.art_max_len,
#         pad_to_max_length=True,
#         truncation=True,
#         padding="max_length"
#         )
    
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     return (
#         torch.tensor(input_ids, dtype=torch.long),
#         torch.tensor(attention_mask, dtype=torch.long)
#         )

#   def post_process(self, generated_ids):
#     preds = [
#         self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         for generated_id in generated_ids
#         ]
# #     return " ".join(preds)
# # model = CNNDailyMailModel()
# # model = model.to(Config.DEVICE)

# # def get_important_paragh(article_text):
# #   data = Process(article_text)
# #   input_ids, attention_mask = data.pre_process()
# #   input_ids = input_ids.to(Config.DEVICE)
# #   attention_mask = attention_mask.to(Config.DEVICE)

# #   with torch.no_grad():
# #     generated_ids = model.model.generate(
# #         input_ids = input_ids,
# #         attention_mask = attention_mask,
# #         max_length=150,
# #         num_beams=2,
# #         repetition_penalty=2.5,
# #         length_penalty=1.0,
# #         early_stopping=True
# #     )

# #   predicted_high = data.post_process(generated_ids)
# #   return predicted_high


# # MODEL = torch.load("sarath_modell.torch")

# # # load("lol adb fasub")
# # # print(type(load))




# # model("cjandcvkjabc c lkjcbwekjc wkjecbwkec lwcbekj ")
#-------------------------------------
# import torch


# # Define a function to load the model
# def load_model(model_class, model_path, device):
#     model = model_class()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     print("Model loaded successfully.")
#     return model

# # Define a function to predict using the loaded model
# def predict(model, article_text, device):
#     data = Process(article_text)
#     input_ids, attention_mask = data.pre_process()
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)

#     with torch.no_grad():
#         generated_ids = model.model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_length=150,
#             num_beams=2,
#             repetition_penalty=2.5,
#             length_penalty=1.0,
#             early_stopping=True
#         )

#     predicted_high = data.post_process(generated_ids)
#     return predicted_high

# # Example usage:
# # Save the model
# # save_model(model, "saved_model.pt")

# # Load the model
# loaded_model = load_model(CNNDailyMailModel, "saved_model.pt", Config.DEVICE)

# # Example article text for prediction
# article_text = """
#           The tower is 324 metres (1,063 ft) tall, about the same height as an
#           81-storey building, and the tallest structure in Paris. Its base is
#           square, measuring 125 metres (410 ft) on each side. During its construction,
#           the Eiffel Tower surpassed the Washington Monument to become the tallest
#           man-made structure in the world, a title it held for 41 years until the Chrysler
#           Building in New York City was finished in 1930. It was the first structure to
#           reach a height of 300 metres. Due to the addition of a broadcasting aerial at
#           the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2
#           metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest
#           free-standing structure in France after the Millau Viaduct.
#        """
# # print(loaded_model)
# # Predict using the loaded model
# predicted_highlights = predict(loaded_model, article_text, Config.DEVICE)
# print("Predicted Highlights:")
# print(predicted_highlights)
#--------------------------------------------------------------
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Config:
    EPOCHS = 4
    MAX_LEN = 512
    HIG_MAX_LEN = 64
    ART_MAX_LEN = 512
    LEARNING_RATE = 3e-5
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT = "t5-base"
    MODEL_PATH = "t5ass.bin"

    TOKENIZER = T5Tokenizer.from_pretrained(CHECKPOINT)
    MODEL = T5ForConditionalGeneration.from_pretrained(CHECKPOINT, return_dict=True)

class CNNDailyMailModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Config.MODEL

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output.loss, output.logits

class Process:
    def __init__(self, article_text):
        self.article_text = article_text
        self.art_max_len = Config.ART_MAX_LEN
        self.tokenizer = Config.TOKENIZER

    def pre_process(self):
        article_text = str(self.article_text)
        article_text = " ".join(article_text.split())

        inputs = self.tokenizer.batch_encode_plus(
            [article_text],
            max_length=self.art_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )

    def post_process(self, generated_ids):
        preds = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return " ".join(preds)

# Define a function to load the model
def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded successfully.")
    return model

# Define a function to predict using the loaded model
def predict(model, article_text, device):
    print("start")
    data = Process(article_text)
    input_ids, attention_mask = data.pre_process()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

    predicted_high = data.post_process(generated_ids)
    print(predicted_high)
    return predicted_high

#Example usage:
# Load the model
loaded_model = load_model(CNNDailyMailModel, "siva_model.pt", Config.DEVICE)

# Example article text for prediction
article_text = """
          Batman[a] is a superhero appearing in American comic books published by DC Comics. The character was created by artist Bob Kane and writer Bill Finger, and debuted in the 27th issue of the comic book Detective Comics on March 30, 1939. In the DC Universe continuity, Batman is the alias of Bruce Wayne, a wealthy American playboy, philanthropist, and industrialist who resides in Gotham City. Batman's origin story features him swearing vengeance against criminals after witnessing the murder of his parents Thomas and Martha as a child, a vendetta tempered with the ideal of justice. He trains himself physically and intellectually, crafts a bat-inspired persona, and monitors the Gotham streets at night. Kane, Finger, and other creators accompanied Batman with supporting characters, including his sidekicks Robin and Batgirl; allies Alfred Pennyworth, James Gordon, and Catwoman; and foes such as the Penguin, the Riddler, Two-Face, and his archenemy, the Joker.

Kane conceived Batman in early 1939 to capitalize on the popularity of DC's Superman; although Kane frequently claimed sole creation credit, Finger substantially developed the concept from a generic superhero into something more bat-like. The character received his own spin-off publication, Batman, in 1940. Batman was originally introduced as a ruthless vigilante who frequently killed or maimed criminals, but evolved into a character with a stringent moral code and strong sense of justice. Unlike most superheroes, Batman does not possess any superpowers, instead relying on his intellect, fighting skills, and wealth. The 1960s Batman television series used a camp aesthetic, which continued to be associated with the character for years after the show ended. Various creators worked to return the character to his darker roots in the 1970s and 1980s, culminating with the 1986 miniseries The Dark Knight Returns by Frank Miller.

DC has featured Batman in many comic books, including comics published under its imprints such as Vertigo and Black Label. The longest-running Batman comic, Detective Comics, is the longest-running comic book in the United States. Batman is frequently depicted alongside other DC superheroes, such as Superman and Wonder Woman, as a member of organizations such as the Justice League and the Outsiders. In addition to Bruce Wayne, other characters have taken on the Batman persona on different occasions, such as Jean-Paul Valley / Azrael in the 1993–1994 "Knightfall" story arc; Dick Grayson, the first Robin, from 2009 to 2011; and Jace Fox, son of Wayne's ally Lucius, as of 2021.[4] DC has also published comics featuring alternate versions of Batman, including the incarnation seen in The Dark Knight Returns and its successors, the incarnation from the Flashpoint (2011) event, and numerous interpretations from Elseworlds stories."""

# Predict using the loaded model
predicted_highlights = predict(loaded_model, article_text, Config.DEVICE)
print("Predicted Highlights:")
#print(predicted_highlights)
# import requests
# from bs4 import BeautifulSoup

# def get_url_paragraphs(url):
#   response = requests.get(url)
#   soup = BeautifulSoup(response.text, 'html.parser')
#   # Find all <p> tags and extract their text
#   paragraphs = [p.get_text() for p in soup.find_all('p')]
#   return " ".join(paragraphs)

# import gradio as gr
# import speech_recognition as sr

# def transcribe_audio(audio_file_path):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file_path) as source:
#         audio_data = recognizer.record(source)
#         text = recognizer.recognize_google(audio_data)
#         return text

# def summarize_text(input_text=None, url=None, audio_file=None):
#     if audio_file is not None:
#         # Transcribe the audio file to text
#         input_text = transcribe_audio(audio_file)

#     if url:
#         # Extract text from the URL
#         input_text = get_url_paragraphs(url)

#     if input_text:
#         # Summarize the extracted or direct input text
#         return predict(loaded_model, input_text, Config.DEVICE)
#     else:
#         return "No text provided for summarization."

# interface = gr.Interface(
#     fn=summarize_text,
#     inputs=[
#         gr.Textbox(lines=7, placeholder="Enter text here..."),
#         gr.Textbox(label="Or enter a URL", placeholder="Enter URL here..."),
#         gr.Audio(label="Or upload an audio file for transcription and summarization")
#     ],
#     outputs=gr.Textbox(),
#     title="Text & Audio Summarization",
#     description="Summarize the provided text, the contents of a given URL, or spoken content from an audio file."
# )

# interface.launch()

