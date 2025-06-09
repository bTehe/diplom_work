# app/utils/reports.py
"""
Модуль для роботи зі звітами інференсу.

Цей модуль містить утиліти для:
- Пошуку CSV-файлів з результатами інференсу.
- Переліку наявних CSV та PDF звітів.
- Формування підсумкових DataFrame за результатами інференсу.
- Збереження детальних звітів у форматах PDF та CSV.

Сутності:
    logger (logging.Logger) – логер для повідомлень та відладки.
    list_inference_results() – функція для переліку CSV-файлів з результатами.
    list_reports() – функція для переліку існуючих звітів.
    generate_summary_df() – генерація DataFrame з підсумками одного файлу результатів.
    save_pdf_report() – збереження звіту у PDF.
    save_csv_report() – збереження звіту у CSV.
"""

import glob
import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .logging_config import setup_logger

# Ініціалізація логера
logger: logging.Logger = setup_logger('reports', 'reports')

# Реєстрація шрифту з підтримкою української
FONT_PATH = Path(__file__).parent / 'fonts' / 'DejaVuSans.ttf'
pdfmetrics.registerFont(TTFont('DejaVuSans', str(FONT_PATH)))

# Підготовка стилів для PDF
_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle(
    name='UkrTitle',
    parent=_styles['Title'],
    fontName='DejaVuSans',
    leading=24,
))
_styles.add(ParagraphStyle(
    name='UkrBody',
    parent=_styles['Normal'],
    fontName='DejaVuSans',
    fontSize=12,
    leading=16,
))


def list_inference_results(incoming_dir: Union[str, Path]) -> List[str]:
    """
    Перелічує CSV-файли з результатами інференсу у теці.

    Шукає файли з суфіксом '_results.csv', сортує їх за назвою.

    Args:
        incoming_dir (Union[str, Path]): Шлях до теки з результатами.

    Returns:
        List[str]: Список імен файлів, що закінчуються на '_results.csv'.
    """
    incoming_path = Path(incoming_dir)
    try:
        pattern = str(incoming_path / '*_results.csv')
        files = sorted(Path(p).name for p in glob.glob(pattern))
        logger.info(
            "Знайдено %d файлів результатів у %s",
            len(files),
            incoming_path,
        )
        return files
    except Exception as exc:
        logger.error(
            "Помилка переліку файлів результатів у %s: %s",
            incoming_path,
            exc,
        )
        return []


def list_reports(report_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Перелічує наявні CSV та PDF звіти у теці.

    Args:
        report_dir (Union[str, Path]): Шлях до теки зі звітами.

    Returns:
        Dict[str, List[str]]: Словник з ключами 'csv' та 'pdf', 
            кожен містить список відповідних імен файлів.
    """
    base = Path(report_dir)
    try:
        all_files = [p.name for p in base.iterdir() if p.is_file()]
        csv_files = sorted(f for f in all_files if f.lower().endswith('.csv'))
        pdf_files = sorted(f for f in all_files if f.lower().endswith('.pdf'))
        logger.info(
            "Знайдено %d CSV та %d PDF звітів у %s",
            len(csv_files),
            len(pdf_files),
            base,
        )
        return {'csv': csv_files, 'pdf': pdf_files}
    except Exception as exc:
        logger.error(
            "Помилка переліку звітів у %s: %s",
            base,
            exc,
        )
        return {'csv': [], 'pdf': []}


def generate_summary_df(incoming_dir: Union[str, Path],
                        result_csv: str) -> pd.DataFrame:
    """
    Формує однорядковий DataFrame із підсумковими метриками.

    Метрики: загальна кількість рядків, кількість помилок,
    рівень помилок, середня впевненість.

    Args:
        incoming_dir (Union[str, Path]): Шлях до теки з CSV-файлами.
        result_csv (str): Ім'я файлу з результатами.

    Returns:
        pd.DataFrame: DataFrame зі стовпцями
            ['file', 'total_rows', 'errors', 'error_rate', 'avg_confidence'].

    Raises:
        Exception: У разі помилки під час обробки файлу.
    """
    path = Path(incoming_dir) / result_csv
    try:
        df = pd.read_csv(path)
        total = len(df)
        errors = int((df['pred'] != df['true']).sum())
        avg_conf = (float(df['confidence'].mean())
                    if not df['confidence'].isna().all() else 0.0)

        summary = pd.DataFrame([{
            'file': result_csv,
            'total_rows': total,
            'errors': errors,
            'error_rate': round(errors / total, 4) if total > 0 else 0.0,
            'avg_confidence': round(avg_conf, 4),
        }])

        logger.info(
            "Згенеровано підсумок для %s: %d рядків, %d помилок",
            result_csv,
            total,
            errors,
        )
        return summary

    except Exception as exc:
        logger.error(
            "Помилка генерації підсумку для %s: %s",
            result_csv,
            exc,
        )
        raise


def save_pdf_report(incoming_dir: Union[str, Path],
                    report_dir: Union[str, Path],
                    result_csv: str) -> str:
    """
    Створює PDF-звіт на основі першого рядка метрик CSV-файлу.

    У PDF включено заголовок і ключові метрики з жирними підписами.

    Args:
        incoming_dir (Union[str, Path]): Шлях до теки з CSV-файлом.
        report_dir (Union[str, Path]): Шлях до теки для збереження PDF.
        result_csv (str): Ім'я CSV-файлу результатів.

    Returns:
        str: Ім'я згенерованого PDF-файлу.
    """
    path = Path(incoming_dir) / result_csv
    df = pd.read_csv(path)
    first_row = df.iloc[0]

    pdf_name = f"{Path(result_csv).stem}_report.pdf"
    pdf_path = Path(report_dir) / pdf_name

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    elements = [
        Paragraph("Звіт за результатами інференсу", _styles['UkrTitle']),
        Spacer(1, 12),
    ]

    metric_fields = [
        ('Точність (Accuracy)', 'accuracy'),
        ('Precision (macro)', 'precision'),
        ('Recall (macro)', 'recall'),
        ('F1-score (macro)', 'f1'),
        ('Weighted F1', 'weighted_f1'),
        ('Log-loss', 'log_loss'),
        ('Top-3 accuracy', 'top_3_accuracy'),
        ('Top-5 accuracy', 'top_5_accuracy'),
        ('ROC AUC (micro)', 'roc_auc_micro'),
        ('ROC AUC (macro)', 'roc_auc_macro'),
        ('PR AUC (micro)', 'pr_auc_micro'),
        ('PR AUC (macro)', 'pr_auc_macro'),
        ('Модель', 'model_used'),
        ('Обладнання', 'hardware'),
        ('Час інференсу', 'inference_ts'),
    ]

    for title, column in metric_fields:
        if column in first_row:
            text = f"<b>{title}:</b> {first_row[column]}"
            elements.append(Paragraph(text, _styles['UkrBody']))
            elements.append(Spacer(1, 6))
        else:
            logger.warning(
                "Відсутнє поле '%s' у даних для PDF-звіту %s",
                column,
                result_csv,
            )

    doc.build(elements)
    logger.info("Збережено PDF-звіт: %s", pdf_path)
    return pdf_name


def save_csv_report(incoming_dir: Union[str, Path],
                    report_dir: Union[str, Path],
                    result_csv: str) -> str:
    """
    Створює CSV-звіт з вибраними стовпцями з повного файлу результатів.

    Вихідний файл містить стовпці 'parameter' та 'value'.

    Args:
        incoming_dir (Union[str, Path]): Шлях до теки з CSV-файлом.
        report_dir (Union[str, Path]): Шлях до теки для збереження CSV.
        result_csv (str): Ім'я CSV-файлу результатів.

    Returns:
        str: Ім'я згенерованого CSV-файлу.
    """
    desired_cols = [
        'true', 'pred', 'confidence',
        'accuracy', 'precision', 'recall', 'f1', 'weighted_f1', 'log_loss',
        'top_3_accuracy', 'top_5_accuracy',
        'roc_auc_micro', 'roc_auc_macro',
        'pr_auc_micro', 'pr_auc_macro',
        'model_used', 'hardware', 'inference_ts',
    ]

    path = Path(incoming_dir) / result_csv
    df = pd.read_csv(path)

    missing = [c for c in desired_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Пропущено стовпці %s у %s; вони будуть проігноровані",
            missing,
            result_csv,
        )

    report_df = df[[c for c in desired_cols if c in df.columns]]
    out_name = f"{Path(result_csv).stem}_report.csv"
    out_path = Path(report_dir) / out_name

    report_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    logger.info("Збережено CSV-звіт: %s", out_path)
    return out_name


__all__ = [
    'list_inference_results',
    'list_reports',
    'generate_summary_df',
    'save_pdf_report',
    'save_csv_report',
]