# app/blueprints/data_ingest/routes.py
import os
from flask import (
    Blueprint, render_template,
    request, flash, redirect,
    url_for, current_app
)
from werkzeug.utils import secure_filename
from ...utils.data_loader import load_csv_preview

# Ініціалізація Blueprint
data_ingest_bp = Blueprint(
    'data_ingest',
    __name__,
    template_folder='templates',
    url_prefix='/data-ingest'
)

# Дозволені формати файлів
ALLOWED_EXTENSIONS = {'csv', 'pcap'}

def allowed_file(filename: str) -> bool:
    """
    Перевіряє розширення файлу
    """
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@data_ingest_bp.route('/', methods=['GET', 'POST'])
def upload():
    """
    Обробка завантаження файлу та генерація прев'ю
    """
    preview = None

    if request.method == 'POST':
        file = request.files.get('file')
        # Перевірка наявності файлу
        if not file or file.filename == '':
            flash('Не обрано файл для завантаження.', 'warning')
            return redirect(request.url)

        # Перевірка формату
        if not allowed_file(file.filename):
            flash('Недопустимий формат файлу. Дозволено лише CSV або PCAP.', 'danger')
            return redirect(request.url)

        # Безпечне ім'я та шлях збереження
        filename = secure_filename(file.filename)
        upload_dir = current_app.config.get('UPLOAD_FOLDER')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)

        try:
            # Збереження файлу
            file.save(file_path)
            flash(f'Файл "{filename}" успішно завантажено.', 'success')

            # Генерація прев'ю перших 5 рядків
            preview = load_csv_preview(file_path, n=5)
        except Exception as e:
            flash(f'Помилка під час збереження файлу: {e}', 'danger')

    return render_template(
        'data_ingest/upload.html',
        preview=preview
    )