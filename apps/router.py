# Created by guxu at 10/24/24
import time

from . import app
from flask import request, jsonify
from .services.text_summary_service import TextSummaryService
from .services.speech_to_text_service import SpeechToTextService

@app.route('/')
def index():
    return "OK"

@app.route('/summarize', methods=['POST'])
def summarize():

    start_time = time.time()
    file = request.files["file"]
    end1 = time.time()
    transfer_time = end1 - start_time

    speech_to_text_service = SpeechToTextService()
    text = speech_to_text_service.speech_to_text(file)
    end2 = time.time()
    s2t_time = end2 - end1

    text_summary_service = TextSummaryService()
    summary = text_summary_service.summarize_text(text)
    end3 =  time.time()
    t2s_time = end3 - end2

    response = {
        "data": summary,
        "metaData": {
            "totalTime": end3 - start_time,
            "transferTime": transfer_time,
            "speechToTextTime": s2t_time,
            "textToSummaryTime": t2s_time
        }
    }
    return jsonify(response)