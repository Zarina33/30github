# api_local.py
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import zipfile
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize the model at startup
print("Initializing Whisper model (this will download the model if it's not cached)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model is ready for transcription!")

ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

def allowed_audio_file(filename):
    # Skip macOS hidden files and folders
    if filename.startswith('._') or filename.startswith('.'):
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'text': '', 'error': 'No file provided'}), 400
    
    zip_file = request.files['file']
    if zip_file.filename == '':
        return jsonify({'text': '', 'error': 'No file selected'}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and extract zip file
            zip_path = os.path.join(temp_dir, secure_filename(zip_file.filename))
            zip_file.save(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only audio files, skip __MACOSX
                for file_info in zip_ref.filelist:
                    filename = file_info.filename
                    if not filename.startswith('__MACOSX') and allowed_audio_file(os.path.basename(filename)):
                        zip_ref.extract(file_info, temp_dir)
            
            # Process audio files
            transcriptions = []
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    if allowed_audio_file(filename):
                        filepath = os.path.join(root, filename)
                        try:
                            # Transcribe with Whisper
                            segments, info = model.transcribe(filepath, beam_size=5)
                            
                            # Combine all segments
                            full_text = ' '.join(segment.text for segment in segments)
                            
                            transcriptions.append({
                                'filename': filename,
                                'text': full_text,
                                'language': info.language,
                                'duration': info.duration
                            })
                        except Exception as e:
                            print(f"Error processing {filename}: {str(e)}")
                            continue
            
            if not transcriptions:
                return jsonify({'text': '', 'error': 'No valid audio files found'}), 400
            
            combined_text = ' '.join(t['text'] for t in transcriptions)
            return jsonify({
                'text': combined_text,
                'files': transcriptions
            })
            
    except zipfile.BadZipFile:
        return jsonify({'text': '', 'error': 'Invalid ZIP file'}), 400
    except Exception as e:
        return jsonify({'text': '', 'error': f'Error: {str(e)}'}), 500

@app.route('/')
def index():
    return "Whisper Transcription API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=8000)