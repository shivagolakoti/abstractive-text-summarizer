from flask import Flask, render_template, request, jsonify
from lol import CNNDailyMailModel, load_model, predict, Config
import requests
from bs4 import BeautifulSoup
import docx2txt
import PyPDF2
import io
from PyPDF2 import PdfReader
import speech_recognition as sr

app = Flask(__name__)

# Load the pre-trained model
loaded_model = load_model(CNNDailyMailModel, "siva_model.pt", Config.DEVICE)

# Function to transcribe audio to text
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

# Function to extract text from a document file
def extract_text(file):
    filename = file.filename
    if filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        raise ValueError("Unsupported file format. Only .docx and .pdf files are supported.")

def extract_text_from_docx(docx_file_path):
    text = docx2txt.process(docx_file_path)
    return text

def extract_text_from_pdf(pdf_file):
    text = ''
    pdf_reader = PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to get text from a URL
def get_url_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return " ".join(paragraphs)

@app.route('/')
def home():
    return render_template('happy2.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text_input = request.form['textInput']
    url_input = request.form['urlInput']
    document_file = request.files['documentInput']
    audio_file = request.files['audioInput']
    
    if text_input:
        summary = predict(loaded_model, text_input, Config.DEVICE)
    elif url_input:
        url_text = get_url_text(url_input)
        summary = predict(loaded_model, url_text, Config.DEVICE)
    elif document_file:
        document_text =extract_text(document_file)
        summary = predict(loaded_model, document_text, Config.DEVICE)
    elif audio_file:
        audio_text = transcribe_audio(audio_file)
        summary = predict(loaded_model, audio_text, Config.DEVICE)

    return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
