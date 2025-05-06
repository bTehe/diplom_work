# app/blueprints/main/routes.py
import os
from flask import Blueprint, render_template, current_app

main_bp = Blueprint(
    'main', __name__,
    template_folder='templates',
    url_prefix='/'
)

@main_bp.route('/')
def dashboard():
    cfg = current_app.config

    stats = {
        'raw_files':        len(os.listdir(cfg['UPLOAD_FOLDER'])),
        'processed_files':  len(os.listdir(cfg['PROCESSED_FOLDER'])),
        'models':           len([f for f in os.listdir(cfg['MODEL_FOLDER']) if f.endswith('.h5')]),
        'inference_runs':   len(os.listdir(cfg['INFERENCE_FOLDER'])),
        'reports':          len(os.listdir(cfg['REPORT_FOLDER'])),
    }

    return render_template('main/dashboard.html', stats=stats)
