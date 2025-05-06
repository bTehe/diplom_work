import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Загальні налаштування
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_FOLDER  = os.path.join(BASE_DIR, 'models')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
INFERENCE_FOLDER = os.path.join(BASE_DIR, 'data', 'inference_results')
REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')

os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(INFERENCE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)