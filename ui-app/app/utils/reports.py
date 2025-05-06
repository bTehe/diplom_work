# app/utils/reports.py
import os
import pandas as pd
from glob import glob
from flask import current_app
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def list_inference_results(inc_dir: str) -> list[str]:
    """Повертає список CSV із результатами inference."""
    pattern = os.path.join(inc_dir, "*_results.csv")
    return sorted([os.path.basename(p) for p in glob(pattern)])

def list_reports(report_dir: str) -> dict:
    """Повертає два списки: CSV- та PDF-звіти."""
    files = os.listdir(report_dir)
    return {
        'csv': sorted(f for f in files if f.lower().endswith('.csv')),
        'pdf': sorted(f for f in files if f.lower().endswith('.pdf'))
    }

def generate_summary_df(result_csv: str) -> pd.DataFrame:
    """
    Збирає з одного result_csv статистику:
      – total_rows, errors_count, error_rate, avg_confidence
    """
    path = os.path.join(current_app.config['INFERENCE_FOLDER'], result_csv)
    df = pd.read_csv(path)
    total = len(df)
    errors = (df['pred'] != df['true']).sum()
    avg_conf = df['confidence'].mean()
    summary = pd.DataFrame([{
        'file': result_csv,
        'total_rows': total,
        'errors': errors,
        'error_rate': round(errors/total, 4),
        'avg_confidence': round(avg_conf, 4)
    }])
    return summary

def save_csv_report(result_csv: str) -> str:
    """Зберігає single-line CSV-звіт в REPORT_FOLDER, повертає ім’я файлу."""
    df_sum = generate_summary_df(result_csv)
    fname = f"{os.path.splitext(result_csv)[0]}_report.csv"
    out_path = os.path.join(current_app.config['REPORT_FOLDER'], fname)
    df_sum.to_csv(out_path, index=False)
    return fname

def save_pdf_report(result_csv: str) -> str:
    """
    Генерує PDF-звіт із таблицею summary та описом, повертає ім’я PDF.
    Використовує reportlab.
    """
    df_sum = generate_summary_df(result_csv)
    pdf_name = f"{os.path.splitext(result_csv)[0]}_report.pdf"
    pdf_path = os.path.join(current_app.config['REPORT_FOLDER'], pdf_name)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("Звіт за результатами inference", styles['Title']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"Файл: {result_csv}", styles['Normal']))
    elems.append(Spacer(1, 12))

    # Створюємо таблицю з заголовком і даними
    data = [df_sum.columns.tolist()] + df_sum.values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ]))
    elems.append(table)

    doc.build(elems)
    return pdf_name
