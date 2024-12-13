# Created by guxu at 10/24/24
import os
from flask import Flask
from flask_cors import CORS

PROJECT_BASE_PATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

CORS(app)

import apps.router