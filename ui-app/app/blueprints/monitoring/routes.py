# app/blueprints/monitoring/routes.py
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
from ...utils.monitoring import list_incident_files, load_incidents, filter_incidents

# Ініціалізація Blueprint
monitoring_bp = Blueprint(
    'monitoring',
    __name__,
    template_folder='templates',
    url_prefix='/monitoring'
)

@monitoring_bp.route('/', methods=['GET', 'POST'])
def monitor():
    """
    Сторінка відображення та фільтрації інцидентів з результатів inference.
    """
    # Директорія з результатами inference
    inc_dir = current_app.config['INFERENCE_FOLDER']

    # Список доступних файлів інцидентів
    files = list_incident_files(inc_dir)

    table_html = None
    selected = {
        'file': None,
        'pred_class': None,
        'min_conf': 0.0
    }

    if request.method == 'POST':
        # Отримуємо параметри форми
        fname = request.form.get('incident_file')
        pred_class = request.form.get('pred_class')
        min_conf = request.form.get('min_confidence') or "0"

        # Валідація
        if not fname:
            flash('Будь ласка, оберіть файл із інцидентами.', 'warning')
            return redirect(url_for('monitoring.monitor'))
        if fname not in files:
            flash(f"Файл '{fname}' не знайдено.", 'danger')
            return redirect(url_for('monitoring.monitor'))

        # Зберігаємо вибране
        selected['file'] = fname
        selected['pred_class'] = int(pred_class) if pred_class != '' else None
        selected['min_conf'] = float(min_conf)

        try:
            # Завантажуємо та фільтруємо інциденти
            df = load_incidents(fname, inc_dir)
            df_filt = filter_incidents(
                df,
                pred_class=selected['pred_class'],
                min_confidence=selected['min_conf']
            )
            table_html = df_filt.to_html(
                classes="table table-striped table-hover",
                index=False
            )
        except Exception as e:
            flash(f'Помилка при завантаженні інцидентів: {e}', 'danger')
            return redirect(url_for('monitoring.monitor'))

    # Рендеримо шаблон
    return render_template(
        'monitoring.html',
        files=files,
        table_html=table_html,
        selected=selected
    )
