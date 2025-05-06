# app/utils/data_loader.py
import pandas as pd

def load_csv_preview(path: str, n: int = 5) -> str:
    """
    Зчитує перші n рядків CSV і повертає їх у вигляді HTML-таблиці.
    """
    try:
        df = pd.read_csv(path, nrows=n)
        # повертаємо HTML з Bootstrap-класами для оформлення
        return df.to_html(classes="table table-sm table-striped", index=False)
    except Exception as e:
        return f"<div class='alert alert-danger'>Помилка при завантаженні превʼю: {e}</div>"
