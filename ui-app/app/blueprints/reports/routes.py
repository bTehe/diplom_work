# app/blueprints/reports/routes.py

import os
from flask import (
    Blueprint,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    send_from_directory,
    current_app
)
from ...utils.reports import (
    list_inference_results,
    list_reports,
    save_csv_report,
    save_pdf_report
)

reports_bp = Blueprint(
    'reports',
    __name__,
    template_folder='templates',
    url_prefix='/reports'
)

@reports_bp.route('/', methods=['GET', 'POST'])
def reports():
    """
    Сторінка генерації та перегляду звітів (CSV/PDF).
    """
    inc_dir    = current_app.config['INFERENCE_FOLDER']
    report_dir = current_app.config['REPORT_FOLDER']

    # Файли результатів inference та вже існуючі звіти
    results  = list_inference_results(inc_dir)
    existing = list_reports(report_dir)
    generated = None

    if request.method == 'POST':
        fname = request.form.get('result_file')
        rtype = request.form.get('report_type')

        # Валідуємо вибір
        if not fname or not rtype:
            flash('Будь ласка, оберіть файл і тип звіту.', 'warning')
            return redirect(url_for('reports.reports'))

        try:
            # Генеруємо звіт
            if rtype == 'csv':
                generated = save_csv_report(fname)
            else:
                generated = save_pdf_report(fname)

            flash(f'Звіт "{generated}" успішно згенеровано.', 'success')
        except Exception as e:
            flash(f'Помилка при генерації звіту: {e}', 'danger')
            return redirect(url_for('reports.reports'))

        # Оновлюємо список
        existing = list_reports(report_dir)

    return render_template(
        'reports.html',
        results=results,
        existing=existing,
        generated=generated
    )

@reports_bp.route('/download/<path:filename>')
def download_report(filename):
    """
    Віддає файл звіту (CSV або PDF) для завантаження.
    """
    report_dir = current_app.config['REPORT_FOLDER']
    return send_from_directory(
        report_dir,
        filename,
        as_attachment=True
    )
