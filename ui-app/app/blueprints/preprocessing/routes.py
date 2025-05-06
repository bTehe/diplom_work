# app/blueprints/preprocessing/routes.py
import os
from flask import (
    Blueprint,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    current_app
)
from ...utils.preprocessing import list_raw_files, preprocess_file

# Ініціалізація Blueprint
preprocess_bp = Blueprint(
    'preprocessing',
    __name__,
    template_folder='templates',
    url_prefix='/preprocessing'
)

@preprocess_bp.route('/', methods=['GET', 'POST'])
def preprocess():
    """
    Сторінка для вибору raw CSV, запуску preprocessing та EDA
    """
    # Шляхи до каталогів
    raw_dir       = current_app.config['UPLOAD_FOLDER']
    processed_dir = current_app.config['PROCESSED_FOLDER']

    # Зчитуємо список доступних файлів
    files = list_raw_files(raw_dir)
    summary = None

    if request.method == 'POST':
        filename = request.form.get('filename')
        # Валідація вибору
        if not filename:
            flash('Будь ласка, оберіть файл для обробки.', 'warning')
            return redirect(url_for('preprocessing.preprocess'))
        if filename not in files:
            flash('Обраного файлу не знайдено.', 'danger')
            return redirect(url_for('preprocessing.preprocess'))

        try:
            # Виконати preprocessing
            summary = preprocess_file(
                filename=filename,
                raw_dir=raw_dir,
                processed_dir=processed_dir
            )
            flash(f'Файл "{filename}" успішно оброблено.', 'success')
        except Exception as e:
            flash(f'Помилка під час обробки: {e}', 'danger')
            return redirect(url_for('preprocessing.preprocess'))

    # Відобразити сторінку з формою та результатом
    return render_template(
        'preprocessing.html',
        files=files,
        summary=summary
    )
