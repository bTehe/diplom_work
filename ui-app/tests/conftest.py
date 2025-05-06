import pytest
import os
import tempfile
from app import create_app

@pytest.fixture
def client():
    # налаштовуємо тимчасову instance-папку
    instance_dir = tempfile.mkdtemp()
    app = create_app()
    app.instance_path = instance_dir
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER']    = os.path.join(instance_dir, 'raw')
    app.config['PROCESSED_FOLDER'] = os.path.join(instance_dir, 'processed')
    app.config['MODEL_FOLDER']     = os.path.join(instance_dir, 'models')
    app.config['INFERENCE_FOLDER'] = os.path.join(instance_dir, 'inference_results')
    app.config['REPORT_FOLDER']    = os.path.join(instance_dir, 'reports')

    # створимо папки
    for d in ('UPLOAD_FOLDER','PROCESSED_FOLDER','MODEL_FOLDER','INFERENCE_FOLDER','REPORT_FOLDER'):
        os.makedirs(app.config[d], exist_ok=True)

    with app.test_client() as client:
        yield client
