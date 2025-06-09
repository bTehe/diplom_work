from pathlib import Path
from typing import Dict, List
import os
from .logging_config import setup_logger

# Ініціалізація логера
logger = setup_logger('config', 'config')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Загальні налаштування
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_FOLDER  = os.path.join(BASE_DIR, 'models')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'data', 'processed')
INFERENCE_FOLDER = os.path.join(BASE_DIR, 'data', 'inference_results')
REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')

WINDOW_SIZE = 20

# Створення директорій
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(INFERENCE_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    logger.info("Успішно створено всі необхідні директорії")
except Exception as e:
    logger.error(f"Помилка при створенні директорій: {str(e)}")

# # 1) log1p: сильна права асиметрія (skew > 1)
# log1p_features = [
#     'Active Mean',
#     'Average Packet Size',
#     'Bwd Header Length',
#     'Bwd IAT Mean',
#     'Bwd IAT Std',
#     'Down/Up Ratio',
#     'Flow Bytes/s',
#     'Flow Duration',
#     'Flow IAT Mean',
#     'Flow IAT Min',
#     'Flow IAT Std',
#     'Flow Packets/s',
#     'Fwd Header Length',
#     'Fwd IAT Mean',
#     'Fwd IAT Std',
#     'Fwd PSH Flags',
#     'Fwd Packets/s',
#     'Idle Mean',
#     'Max Packet Length',
#     'Min Packet Length',
#     'Packet Length Mean',
#     'Packet Length Std',
#     'Packet Length Variance'
# ]

# # 2) Yeo–Johnson: сильна ліва асиметрія (skew < –1), помірна права асиметрія (0.5 < skew ≤ 1), або наявність нулів/від'ємних
# yeo_johnson_features = [
#     'Total Length of Fwd Packets'
#     'Fwd Packet Length Max'
# ]

# # 3) QuantileTransformer: надзвичайно важкі хвости (skew > 50)
# quantile_transform_features = [
#     'Bwd Packets/s',
#     'Subflow Bwd Packets',
#     'Subflow Fwd Packets',
#     'Total Backward Packets',
#     'Total Fwd Packets'
# ]

# # 4) За рештою ознак (|skew| ≤ 0.5) достатньо RobustScaler або залишити без трансформації:
# robust_features = [
#     'Bwd Avg Bytes/Bulk',
#     'Bwd URG Flags',
#     'Flow IAT Max',
#     'Fwd Avg Bytes/Bulk',
#     'Fwd Avg Packets/Bulk',
#     'Fwd IAT Min',
#     'Fwd Packet Length Min',
# ]

# features_to_bin = [
#     'Flow IAT Min',
#     'Fwd Header Length',
#     'Fwd PSH Flags',
#     'Packet Length Variance',
#     'Total Backward Packets',
#     'Total Fwd Packets'
# ]

# features_to_drop = [
#     'Packet Length Std',
#     'Max Packet Length',
#     'Average Packet Size',
#     'Flow IAT Std',
#     'Fwd IAT Mean',
#     'Bwd IAT Mean',
#     'Fwd Header Length',
#     'Bwd Header Length'
# ]

# CATEGORY_LABELS: Dict[str, List[str]] = {
#     'BENIGN': ['BENIGN'],
#     'DoS': ['DDoS', 'DoS slowloris', 'DoS Hulk', 'DoS GoldenEye'],
#     'PortScan': ['PortScan'],
#     'Bot_Infiltration': ['Bot', 'Infiltration'],
#     'Web': ['Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection'],
#     'FTP_SSH_Patator': ['FTP-Patator', 'SSH-Patator'],
#     'Heartbleed': ['Heartbleed'],
# }

# GROUP_FEATURES: Dict[str, List[str]] = {
#     'dos': ['Fwd Packets/s', 'Bwd Packets/s', 'Flow Duration', 'Flow IAT Min', 'Flow IAT Max', 'SYN Flag Count', 'PSH Flag Count'],
#     'portscan': ['SYN Flag Count', 'FIN Flag Count', 'RST Flag Count', 'Total Fwd Packets', 'Total Backward Packets'],
#     'bot_infiltration': ['Flow Duration', 'Fwd IAT Std', 'Bwd IAT Std', 'Fwd PSH Flags', 'Bwd URG Flags', 'Down/Up Ratio'],
#     'web': ['Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance', 'ACK Flag Count', 'Average Packet Size'],
#     'ftp_ssh_patator': ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Active Mean', 'Idle Mean'],
#     'heartbleed': ['Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd IAT Min', 'Total Length of Fwd Packets', 'Packet Length Std'],
# }

# BASE_FEATURES: List[str] = [
#     'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size', 'Down/Up Ratio',
#     'Packet Length Mean', 'Packet Length Std', 'Min Packet Length', 'Max Packet Length',
#     'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
#     'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
#     'Active Mean', 'Idle Mean', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
#     'Label', 'dow', 'hour', 'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos'
# ]

# STD_FEATURES: List[str] = [
#     'Fwd Packet Length Std', 'Bwd Packet Length Std', 'Flow IAT Std',
#     'Fwd IAT Std', 'Bwd IAT Std', 'Packet Length Std', 'Active Std', 'Idle Std'
# ]

# DATE_MAP: Dict[str, str] = {
#     'Monday': '2023-11-06 12:00:00',
#     'Tuesday': '2023-11-07 12:00:00',
#     'Wednesday': '2023-11-08 12:00:00',
#     'Thursday-Morning': '2023-11-09 09:00:00',
#     'Thursday-Afternoon': '2023-11-09 15:00:00',
#     'Friday-Morning': '2023-11-10 09:00:00',
#     'Friday-Afternoon1': '2023-11-10 13:00:00',
#     'Friday-Afternoon2': '2023-11-10 17:00:00',
# }

# label_mapping = {
#     'BENIGN':            0,
#     'DoS':               1,
#     'PortScan':          2,
#     'Bot_Infiltration':  3,
#     'Web':               4,
#     'FTP_SSH_Patator':   5,
#     'Heartbleed':        6,
# }