from flask import Flask, render_template, request, jsonify, session, send_file
from dotenv import load_dotenv
import os
import requests
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
from PIL import Image
import pickle
import tensorflow as tf
from deep_translator import GoogleTranslator
from langdetect import detect
import time
from scipy.sparse import csr_matrix
from datetime import datetime
import hashlib
import uuid
import tempfile
from flask_cors import CORS

# Load environment variables
load_dotenv()
# Add this after load_dotenv()
print(f"✅ GROQ_API_KEY loaded: {'Yes' if ('GROQ_API_KEY') else 'No'}")
print(f"✅ SECRET_KEY loaded: {'Yes' if os.getenv('FLASK_SECRET_KEY') else 'No'}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in environment variables")

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Enhanced language support with Tamil
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi', 
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'ar': 'Arabic',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'ja': 'Japanese'
}

# Global variable for knowledge base
kb_data = None

# Safe Groq API call with retry mechanism
def call_groq_with_retry(model_name, prompt, max_retries=3, initial_wait=10):
    """
    Calls Groq API with automatic retries on rate-limit errors (429).
    """
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:  # Rate limit
                wait_time = initial_wait * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            print(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    return "Error: API failed after multiple retries due to rate limits."

# JSON Storage Functions
def save_kb_to_json(kb_data, json_file_path):
    """
    Save knowledge base to JSON file with proper NumPy type handling
    """
    try:
        if hasattr(kb_data["tfidf_matrix"], 'toarray'):
            tfidf_dense = kb_data["tfidf_matrix"].toarray()
        else:
            tfidf_dense = kb_data["tfidf_matrix"]
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        json_data = {
            "metadata": {
                "version": "2.0",
                "created_at": datetime.now().isoformat(),
                "num_documents": int(len(kb_data["text_chunks"])),
                "vocabulary_size": int(len(kb_data["vectorizer"].vocabulary_)),
                "language_support": ["en", "hi", "ta", "te", "kn", "ml"]
            },
            "text_chunks": kb_data["text_chunks"],
            "vectorizer_data": {
                "vocabulary": convert_numpy_types(dict(kb_data["vectorizer"].vocabulary_)),
                "idf": convert_numpy_types(kb_data["vectorizer"].idf_.tolist()),
                "feature_names": convert_numpy_types(kb_data["vectorizer"].get_feature_names_out().tolist())
            },
            "tfidf_matrix": {
                "shape": [int(x) for x in tfidf_dense.shape],
                "data": convert_numpy_types(tfidf_dense.tolist())
            }
        }
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
        
        print(f"Knowledge base cached to {json_file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving knowledge base: {str(e)}")
        return False

def load_kb_from_json(json_file_path):
    """
    Load knowledge base from JSON file with proper type handling
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        vectorizer = TfidfVectorizer()
        vocab_dict = json_data["vectorizer_data"]["vocabulary"]
        vectorizer.vocabulary_ = {str(k): int(v) for k, v in vocab_dict.items()}
        vectorizer.idf_ = np.array(json_data["vectorizer_data"]["idf"], dtype=np.float64)
        
        tfidf_array = np.array(json_data["tfidf_matrix"]["data"], dtype=np.float64)
        tfidf_matrix = csr_matrix(tfidf_array)
        
        kb_data = {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "text_chunks": json_data["text_chunks"],
            "metadata": json_data.get("metadata", {})
        }
        
        num_docs = json_data.get("metadata", {}).get("num_documents", len(json_data["text_chunks"]))
        print(f"Knowledge base loaded from cache ({num_docs} documents)")
        return kb_data
        
    except Exception as e:
        print(f"Error loading knowledge base from JSON: {str(e)}")
        return None

def load_knowledge_base():
    """
    Load knowledge base with JSON caching
    """
    json_cache_path = "agro_knowledge_cache.json"
    pdf_path = r'C:\Users\Dharukesh M\Desktop\agrobot\DATA1.pdf'
    
    # Try to load from cache first
    if os.path.exists(json_cache_path):
        try:
            if not os.path.exists(pdf_path) or os.path.getmtime(json_cache_path) > os.path.getmtime(pdf_path):
                print("Loading knowledge base from cache...")
                cached_kb = load_kb_from_json(json_cache_path)
                if cached_kb is not None:
                    return cached_kb
        except Exception as e:
            print(f"Cache loading failed: {str(e)}. Processing from PDF...")
    
    # Process PDF if cache loading failed or doesn't exist
    if os.path.exists(pdf_path):
        print("Processing PDF and creating cache...")
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                print("No documents found in PDF")
                return None
            
            text_splitter = CharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)
            text_chunks = [doc.page_content for doc in texts if doc.page_content.strip()]
            
            if not text_chunks:
                print("No text chunks extracted from PDF")
                return None
            
            vectorizer = TfidfVectorizer(
                max_features=5000, 
                stop_words='english',
                min_df=1,
                max_df=0.9
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_chunks)
            except Exception as e:
                print(f"Vectorization failed: {str(e)}")
                return None
            
            kb_data = {
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "text_chunks": text_chunks
            }
            
            if save_kb_to_json(kb_data, json_cache_path):
                print("Knowledge base cached successfully!")
            
            return kb_data
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None
    else:
        print(f"PDF file not found at: {pdf_path}")
        return None

# Initialize knowledge base on startup
print("Loading knowledge base...")
kb_data = load_knowledge_base()

# Helper functions
def get_relevant_documents(question, top_k=5):
    if kb_data is None:
        return []
    
    vectorizer = kb_data["vectorizer"]
    tfidf_matrix = kb_data["tfidf_matrix"]
    text_chunks = kb_data["text_chunks"]
    
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_indices = [i for i in top_indices if similarities[i] > 0.1]
    relevant_docs = [text_chunks[i] for i in relevant_indices]
    return relevant_docs

def translate_text(text, target_lang, source_lang='en'):
    """
    Enhanced translation with Tamil support
    """
    if target_lang == source_lang:
        return text
    
    try:
        if target_lang == 'ta':
            translator = GoogleTranslator(source=source_lang, target='ta')
        else:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        if len(text) > 4000:
            chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
            translated_chunks = []
            for chunk in chunks:
                try:
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Translation chunk error: {str(e)}")
                    translated_chunks.append(chunk)
            return ' '.join(translated_chunks)
        else:
            return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def get_rag_response(question, user_lang):
    try:
        detected_lang = detect(question) if question else 'en'
        
        if detected_lang != 'en':
            english_question = translate_text(question, 'en', detected_lang)
        else:
            english_question = question
        
        relevant_docs = get_relevant_documents(english_question)
        
        if not relevant_docs:
            tamil_context = ""
            if user_lang == 'ta':
                tamil_context = "\nPlease consider Tamil Nadu's agricultural practices, climate, and common crops like rice, sugarcane, cotton, and millets when providing advice."
            
            prompt = f"""You are an agricultural expert assistant specializing in Indian agriculture. 
                     Provide practical, region-appropriate advice with product recommendations and reliable sources.
                     {tamil_context}
                      
                     Question: {english_question}
                     Answer:"""
        else:
            context = "\n".join(relevant_docs)
            tamil_context = ""
            if user_lang == 'ta':
                tamil_context = "\nConsider Tamil Nadu's agricultural conditions: tropical climate, monsoon patterns, major crops (rice, sugarcane, cotton, millets, coconut), and local farming practices."
            
            prompt = f"""You are an expert Agricultural Assistant specializing in Indian agriculture. 
                        Provide practical, sustainable solutions with product recommendations and references.
                        {tamil_context}
                      
                      Context:
                      {context}
                      
                      Question: {english_question}
                      Answer:"""
        
        english_response = call_groq_with_retry('llama-3.1-8b-instant', prompt)
        
        if user_lang != 'en':
            translated_response = translate_text(english_response, user_lang)
            return translated_response
        else:
            return english_response
            
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        return translate_text(error_msg, user_lang)

# Routes
@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'language' not in session:
        session['language'] = 'en'
    
    return render_template('index.html', 
                         languages=LANGUAGES, 
                         current_language=session['language'])

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        language = data.get('language', 'en')
        
        if not question:
            return jsonify({
                'success': False, 
                'error': translate_text('Please enter a question first.', language)
            })
        
        # Update session
        session['language'] = language
        
        # Get response
        response = get_rag_response(question, language)
        
        # Update chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'type': 'user',
            'message': question,
            'timestamp': datetime.now().isoformat()
        })
        session['chat_history'].append({
            'type': 'ai',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        
        return jsonify({
            'success': True,
            'response': response,
            'chat_history': session['chat_history']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        })

@app.route('/voice-to-text', methods=['POST'])
def voice_to_text():
    try:
        language = request.form.get('language', 'en')
        
        # Map language codes for speech recognition
        speech_lang_map = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'kn': 'kn-IN',
            'ml': 'ml-IN',
            'bn': 'bn-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN',
            'pa': 'pa-IN'
        }
        
        speech_lang = speech_lang_map.get(language, 'en-US')
        
        # Check if audio file was uploaded
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            })
        
        audio_file = request.files['audio']
        
        # Save audio file temporarily
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        audio_file.save(temp_audio_path)
        
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=speech_lang)
            
            # Clean up temp file
            os.remove(temp_audio_path)
            
            return jsonify({
                'success': True,
                'text': text
            })
            
        except sr.UnknownValueError:
            os.remove(temp_audio_path)
            return jsonify({
                'success': False,
                'error': translate_text("Sorry, I couldn't understand what you said.", language)
            })
        except sr.RequestError as e:
            os.remove(temp_audio_path)
            return jsonify({
                'success': False,
                'error': translate_text("Sorry, speech recognition service error.", language)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing voice input: {str(e)}'
        })

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        # Map language codes for gTTS
        tts_lang_map = {
            'en': 'en',
            'hi': 'hi',
            'ta': 'ta',
            'te': 'te',
            'kn': 'kn',
            'ml': 'ml',
            'bn': 'bn',
            'mr': 'mr',
            'gu': 'gu'
        }
        
        tts_lang = tts_lang_map.get(language, 'en')
        
        # Truncate text if too long
        if len(text) > 500:
            text = text[:500] + "..."
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_data = audio_buffer.read()
        audio_base64 = base64.b64encode(audio_data).decode()
        
        return jsonify({
            'success': True,
            'audio_base64': audio_base64
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating speech: {str(e)}'
        })

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        language = request.form.get('language', 'en')
        description = request.form.get('description', '')
        
        if not description:
            fallback_msg = "படத்தில் நீங்கள் கவனிக்கும் அறிகுறிகளை விவரிக்கவும்" if language == 'ta' else "Please describe what you observe in the image"
            return jsonify({
                'success': True,
                'analysis': fallback_msg
            })
        
        # Analyze based on description
        question = f"Based on this plant observation: {description}. Please provide detailed agricultural advice including possible causes, treatment options, and prevention measures."
        analysis_result = get_rag_response(question, language)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error analyzing image: {str(e)}'
        })

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        session['chat_history'] = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error clearing history: {str(e)}'
        })

@app.route('/get-examples')
def get_examples():
    language = request.args.get('language', 'en')
    
    examples_by_language = {
        'en': [
            "What are the best crops to grow in summer?",
            "How to treat common plant diseases?", 
            "What are sustainable farming practices?",
            "How to improve soil fertility naturally?",
            "What is crop rotation and its benefits?",
            "How to manage pests without chemicals?"
        ],
        'ta': [
            "கோடையில் பயிரிட சிறந்த பயிர்கள் எவை?",
            "பொதுவான தாவர நோய்களுக்கு எவ்வாறு சிகிச்சை அளிப்பது?",
            "நீடித்த விவசாய நடைமுறைகள் என்ன?",
            "மண் வளத்தை இயற்கையாக எவ்வாறு மேம்படுத்துவது?",
            "பயிர் சுழற்சி என்றால் என்ன மற்றும் அதன் பலன்கள்?",
            "இரசாயனங்கள் இல்லாமல் பூச்சிகளை எவ்வாறு கட்டுப்படுத்துவது?",
            "நெல் பயிரிடுவதற்கான சிறந்த முறைகள் என்ন?",
            "கரும்பு பயிரில் ஏற்படும் நோய்களை எவ்வாறு தடுப்பது?"
        ],
        'hi': [
            "गर्मी में उगाने के लिए सबसे अच्छी फसलें कौन सी हैं?",
            "सामान्य पौधों के रोगों का इलाज कैसे करें?",
            "टिकाऊ खेती के तरीके क्या हैं?",
            "मिट्टी की उर्वरता को प्राकृतिक रूप से कैसे बढ़ाएं?"
        ]
    }
    
    examples = examples_by_language.get(language, examples_by_language['en'])
    
    return jsonify({
        'success': True,
        'examples': examples
    })

@app.route('/test-api')
def test_api():
    try:
        test_response = call_groq_with_retry('llama-3.1-8b-instant', "Hello, this is a test.")
        if not test_response.startswith("Error"):
            return jsonify({
                'success': True,
                'message': 'API connection successful!'
            })
        else:
            return jsonify({
                'success': False,
                'error': test_response
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'API connection failed: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
