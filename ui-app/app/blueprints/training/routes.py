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
from ...utils.training import train_model_on_csv
from ...utils.preprocessing import list_raw_files

# Ініціалізація Blueprint
training_bp = Blueprint(
    'training',
    __name__,
    template_folder='templates',
    url_prefix='/training'
)

@training_bp.route('/', methods=['GET', 'POST'])
def train():
    """
    Сторінка запуску та моніторингу навчання моделі
    """
    # Шляхи до директорій
    processed_dir = current_app.config['PROCESSED_FOLDER']
    model_dir     = current_app.config['MODEL_FOLDER']

    # Отримуємо список оброблених CSV
    files = list_raw_files(processed_dir)
    metrics = None

    if request.method == 'POST':
        # Параметри з форми
        filename   = request.form.get('filename')
        batch_size = request.form.get('batch_size', type=int) or 64
        epochs     = request.form.get('epochs', type=int) or 20

        # Валідація вхідних даних
        if not filename:
            flash('Будь ласка, оберіть файл для навчання.', 'warning')
            return redirect(url_for('training.train'))
        if filename not in files:
            flash('Обраний файл не знайдено.', 'danger')
            return redirect(url_for('training.train'))

        # Інформуємо про старт тренування
        flash(f"Запуск навчання моделі на '{filename}'...", 'info')
        try:
            metrics = train_model_on_csv(
                filename=filename,
                processed_dir=processed_dir,
                model_dir=model_dir,
                batch_size=batch_size,
                epochs=epochs
            )
            flash('Навчання завершено успішно!', 'success')
        except Exception as e:
            flash(f'Помилка під час навчання: {e}', 'danger')
            return redirect(url_for('training.train'))

    # Відображаємо сторінку з оновленими даними
    return render_template(
        'training.html',  # виправлено шлях до шаблону
        files=files,
        metrics=metrics
    )