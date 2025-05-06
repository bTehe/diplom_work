# app/blueprints/inference/routes.py
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
from ...utils.preprocessing import list_raw_files
from ...utils.inference import list_model_files, predict_from_csv

# Ініціалізація Blueprint
inference_bp = Blueprint(
    'inference',
    __name__,
    template_folder='templates',
    url_prefix='/inference'
)

@inference_bp.route('/', methods=['GET', 'POST'])
def infer():
    """
    Сторінка для запуску inference: вибір датасету та моделі, перегляд результатів.
    """
    # Директорії для даних і моделей
    processed_dir = current_app.config['PROCESSED_FOLDER']
    model_dir     = current_app.config['MODEL_FOLDER']

    # Списки доступних файлів
    csv_files   = list_raw_files(processed_dir)
    model_files = list_model_files(model_dir)
    result = None

    if request.method == 'POST':
        csv_fname   = request.form.get('csv_file')
        model_fname = request.form.get('model_file')

        # Валідація вибору
        if not csv_fname or not model_fname:
            flash('Будь ласка, оберіть і датасет, і модель.', 'warning')
            return redirect(url_for('inference.infer'))

        if csv_fname not in csv_files:
            flash(f"Файл '{csv_fname}' не знайдено.", 'danger')
            return redirect(url_for('inference.infer'))
        if model_fname not in model_files:
            flash(f"Модель '{model_fname}' не знайдена.", 'danger')
            return redirect(url_for('inference.infer'))

        try:
            flash(f"Запуск inference на '{csv_fname}' з моделлю '{model_fname}'...", 'info')
            # Викликаємо утиліту передбачення
            result = predict_from_csv(
                filename=csv_fname,
                processed_dir=processed_dir,
                model_dir=model_dir,
                top_n=5
            )
            flash('Inference виконано успішно!', 'success')
        except Exception as e:
            flash(f'Помилка під час inference: {e}', 'danger')
            return redirect(url_for('inference.infer'))

    return render_template(
        'inference.html',
        csv_files=csv_files,
        model_files=model_files,
        result=result
    )
