from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import nltk
from googletrans import Translator
import concurrent.futures
import time
import langdetect
import requests.exceptions

print("\nĐang cấu hình ứng dụng...")

MODEL_NAME = "facebook/bart-large-cnn"
DATABASE_URI = 'sqlite:///summaries.db'

print(f"\nCUDA khả dụng: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    bart_tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    bart_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    bart_model.eval()
except Exception as e:
    print(f"Lỗi khi tải mô hình BART: {e}")
    exit(1)

print("\nKhởi tạo Google Translate...")
translator = Translator()
if translator is None:
    print("Lỗi: Không thể khởi tạo Translator.")
    exit(1)

try:
    nltk.download('punkt')
except Exception as e:
    print(f"Lỗi khi tải dữ liệu NLTK: {e}")
    exit(1)

print("\nCấu hình Flask và SQLAlchemy...")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

print("\nĐịnh nghĩa model Summarization...")
class Summarization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)

print("\nTạo bảng cơ sở dữ liệu nếu chưa tồn tại...")
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print(f"Lỗi khi tạo bảng cơ sở dữ liệu: {e}")
    exit(1)

def preprocess_text(text):
    """Xóa dòng mới và khoảng trắng thừa."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split())

def split_text_into_token_chunks(text, tokenizer, max_tokens):
    """Chia văn bản thành các chunk dựa trên số lượng token."""
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def detect_language(text):
    """Phát hiện ngôn ngữ bằng langdetect."""
    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        return "unknown"

def get_dynamic_summary_params(text):
    """Xác định tham số tóm tắt dựa trên số lượng từ."""
    word_count = len(text.split())
    if word_count < 200:
        ratio = 0.5
    elif word_count < 500:
        ratio = 0.5
    elif word_count < 700:
        ratio = 0.45
    elif word_count < 900:
        ratio = 0.45
    elif word_count < 1200:
        ratio = 0.45
    elif word_count < 1500:
        ratio = 0.45
    elif word_count < 2000:
        ratio = 0.45
    else:
        ratio = 0.40
    target_length = int(word_count * ratio)
    max_len = max(50, target_length)
    min_len = max(30, int(max_len * 0.8))
    return min_len, max_len

def summarize_chunk(chunk, min_len, max_len):
    """Tóm tắt một đoạn văn bản (chunk) bằng mô hình BART."""
    inputs = bart_tokenizer.encode(
        chunk,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    summary_ids = bart_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=1.0,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.0,
        early_stopping=True
    )

    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def translate_text(text, src, dest):
    """Dịch văn bản bằng Google Translate."""
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except requests.exceptions.ConnectionError:
        print("Lỗi: Không thể kết nối đến máy chủ dịch thuật.")
        return "Lỗi: Không thể kết nối đến máy chủ dịch thuật."
    except Exception as e:
        print(f"Lỗi dịch thuật: {e}")
        return f"Lỗi dịch thuật: {e}"

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    input_text = ""
    status_message = ""
    if request.method == "POST":
        input_text = request.form.get("input_text")
        if input_text:
            status_message = "Processing your input..."
            original_input_text = input_text
            input_text = preprocess_text(input_text)
            input_language = detect_language(input_text)
            if input_language == "vi":
                input_text_for_summarization = translate_text(input_text, 'vi', 'en')
            else:
                input_text_for_summarization = input_text
            min_len, max_len = get_dynamic_summary_params(input_text_for_summarization)
            summarization_prompt = "summarize: " + input_text_for_summarization
            bart_tokens = bart_tokenizer(summarization_prompt, return_tensors="pt", truncation=False)['input_ids'][0]
            if len(bart_tokens) > 1024:
                token_chunks = split_text_into_token_chunks(summarization_prompt, bart_tokenizer, max_tokens=1024)
                summary_parts = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(summarize_chunk, chunk, min_len, max_len) for chunk in token_chunks]
                    for future in concurrent.futures.as_completed(futures):
                        summary_parts.append(future.result())
                summary = " ".join(summary_parts)
            else:
                inputs = bart_tokenizer.encode(summarization_prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
                summary_ids = bart_model.generate(
                    inputs,
                    max_length=max_len,
                    min_length=min_len,
                    length_penalty=1.0,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.0,
                    early_stopping=True
                )
                summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if input_language == "vi":
                summary = translate_text(summary, 'en', 'vi')
            new_entry = Summarization(original_text=original_input_text, summary_text=summary)
            db.session.add(new_entry)
            db.session.commit()
            input_text = original_input_text
            status_message = "Summarization completed!"
    return render_template("index.html", summary=summary, input_text=input_text, status_message=status_message)

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Lỗi khi chạy ứng dụng Flask: {e}")
        exit(1)