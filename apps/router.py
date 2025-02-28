# Created by guxu at 10/24/24
import logging
import time
import json

from . import app
from flask import request, jsonify
from .services.text_summary_service import TextSummaryService
from .services.speech_to_text_service import SpeechToTextService
from .services.file_upload_server import FileUploadService

logging.basicConfig(level=logging.INFO)
@app.route('/')
def index():
    return "OK"

@app.route('/summarize/v1', methods=['POST'])
def summarize():
    start_time = time.time()

    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file in request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        end1 = time.time()
        transfer_time = end1 - start_time

        # audio to text
        speech_to_text_service = SpeechToTextService()
        try:
            text = speech_to_text_service.speech_to_text(file)
        except Exception as e:
            logging.error(f"Speech-to-text processing failed: {e}")
            return jsonify({"error": "Failed to process speech-to-text"}), 500

        end2 = time.time()
        s2t_time = end2 - end1

        # text summarization
        text_summary_service = TextSummaryService()
        try:
            summary = text_summary_service.summarize_text(text)
        except Exception as e:
            logging.error(f"Text summarization failed: {e}")
            return jsonify({"error": "Failed to summarize text"}), 500

        end3 = time.time()
        ts_time = end3 - end2

        response = {
            "data": summary,
            "metaData": {
                "totalTime": end3 - start_time,
                "transferTime": transfer_time,
                "speechToTextTime": s2t_time,
                "textSummarizationTime": ts_time
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file_upload_service = FileUploadService()
        file = request.files["file"]
        metadata = request.form.get('metadata')

        if not file or not metadata:
            logging.error("Missing file or metadata")
            return jsonify({"error": "Missing file or metadata"}), 400

        metadata = json.loads(metadata)
        file_path = file_upload_service.save_file(file, metadata)
        logging.info(f"File saved to: {file_path}")
        return jsonify({"file_path": file_path}), 201
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route('summarize/v2', methods=['POST'])
# def summarizeV2():
#     try:
#         file_path = request.form.get('file_path')
#         text_summary_service = TextSummaryService()
#         summary = text_summary_service.summarize_text(file_path)