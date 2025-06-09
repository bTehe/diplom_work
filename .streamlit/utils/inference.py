# app/utils/inference.py
"""
Модуль для виконання інференсу натренованих моделей, обчислення ключових метрик
та побудови графіків (матриці сплутаності, ROC, PR, Calibration Curve).

Публічний API:
    list_model_files(model_dir) → List[str]
    plot_confusion_matrix(y_true, y_pred, class_names) → str
    plot_roc_curves(y_test_onehot, y_pred_probs, class_names) → str
    plot_pr_curves(y_test_onehot, y_pred_probs, class_names) → str
    plot_calibration(y_test_onehot, y_pred_probs) → str
    infer_on_file(test_file, processed_dir, model_path,
                  window_size, use_gpu, top_k) → Dict[str, Any]
"""

import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from tensorflow.errors import OpError
from tensorflow.python.framework.errors_impl import UnknownError

from .logging_config import setup_logger
from .training import prepare_data, prepare_windowed_data, setup_hardware

# Ініціалізація логера
logger = setup_logger('inference', 'inference')

# Вимкнення інтерактивного бекенду для matplotlib
plt.switch_backend('Agg')
# Відключення XLA JIT для уникнення помилок на GPU
tf.config.optimizer.set_jit(False)

__all__ = [
    'list_model_files',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_calibration',
    'infer_on_file',
]


def list_model_files(model_dir: Union[str, os.PathLike]) -> List[str]:
    """
    Перелічує всі .h5 файли моделей у вказаній теці.

    Args:
        model_dir: шлях до теки з файлами моделей.

    Returns:
        Відсортований список імен файлів з розширенням .h5.
    """
    try:
        files = sorted(
            f for f in os.listdir(model_dir)
            if f.lower().endswith('.h5')
        )
        logger.info("Знайдено %d моделей у %s", len(files), model_dir)
        return files
    except FileNotFoundError:
        logger.error("Не знайдено теку моделей: %s", model_dir)
        return []
    except Exception as exc:
        logger.error("Помилка при переліку моделей: %s", exc)
        return []


def _plot_to_base64(fig: plt.Figure) -> str:
    """
    Конвертує matplotlib.Figure у base64-кодоване PNG-зображення.

    Args:
        fig: об'єкт Figure для збереження.

    Returns:
        Base64-строка PNG-зображення.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Будує матрицю сплутаності та повертає її як base64-зображення.

    Args:
        y_true: масив істинних міток.
        y_pred: масив передбачених міток.
        class_names: назви класів.

    Returns:
        Base64-строка зображення матриці сплутаності.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Передбачена мітка',
        ylabel='Істинна мітка',
        title='Матриця сплутаності',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    return _plot_to_base64(fig)


def plot_roc_curves(
    y_test_onehot: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Будує ROC-криві для кожного класу та обчислює micro/macro AUC.

    Args:
        y_test_onehot: one-hot кодування істинних міток.
        y_pred_probs: передбачені ймовірності по класах.
        class_names: назви класів.

    Returns:
        Base64-строка зображення ROC-кривих.
    """
    present = np.unique(y_test_onehot.argmax(axis=1))
    if present.size < 2:
        logger.warning("Менше двох класів, ROC-криві не побудовані")
        return ''

    fig, ax = plt.subplots()
    for i in present:
        try:
            auc_val = roc_auc_score(
                y_test_onehot[:, i],
                y_pred_probs[:, i]
            )
            fpr, tpr, _ = roc_curve(
                y_test_onehot[:, i],
                y_pred_probs[:, i]
            )
            ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_val:.2f})")
        except Exception:
            continue

    try:
        micro = roc_auc_score(
            y_test_onehot[:, present],
            y_pred_probs[:, present],
            average='micro',
            multi_class='ovr',
            labels=present
        )
        macro = roc_auc_score(
            y_test_onehot[:, present],
            y_pred_probs[:, present],
            average='macro',
            multi_class='ovr',
            labels=present
        )
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax.set_title(f"ROC (micro={micro:.2f}, macro={macro:.2f})")
    except Exception:
        pass

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize='small')
    return _plot_to_base64(fig)


def plot_pr_curves(
    y_test_onehot: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Будує Precision-Recall криві з micro/macro AUC.

    Args:
        y_test_onehot: one-hot кодування істинних міток.
        y_pred_probs: передбачені ймовірності по класах.
        class_names: назви класів.

    Returns:
        Base64-строка зображення PR-кривих.
    """
    present = np.unique(y_test_onehot.argmax(axis=1))
    if present.size < 2:
        logger.warning("Менше двох класів, PR-криві не побудовані")
        return ''

    fig, ax = plt.subplots()
    for i in present:
        try:
            pr_auc = average_precision_score(
                y_test_onehot[:, i],
                y_pred_probs[:, i]
            )
            precision, recall, _ = precision_recall_curve(
                y_test_onehot[:, i],
                y_pred_probs[:, i]
            )
            ax.plot(recall, precision, label=f"{class_names[i]} (AUC={pr_auc:.2f})")
        except Exception:
            continue

    try:
        micro = average_precision_score(
            y_test_onehot[:, present],
            y_pred_probs[:, present],
            average='micro'
        )
        macro = average_precision_score(
            y_test_onehot[:, present],
            y_pred_probs[:, present],
            average='macro'
        )
        ax.set_title(f"PR (micro={micro:.2f}, macro={macro:.2f})")
    except Exception:
        pass

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize='small')
    return _plot_to_base64(fig)


def plot_calibration(
    y_test_onehot: np.ndarray,
    y_pred_probs: np.ndarray
) -> str:
    """
    Будує калибрувальну криву (reliability diagram).

    Args:
        y_test_onehot: one-hot кодування істинних міток.
        y_pred_probs: передбачені ймовірності по класах.

    Returns:
        Base64-строка зображення calibration curve.
    """
    y_true = y_test_onehot.ravel()
    y_prob = y_pred_probs.ravel()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    return _plot_to_base64(fig)


def infer_on_file(
    test_file: str,
    processed_dir: Union[str, os.PathLike],
    model_path: Union[str, os.PathLike],
    window_size: int = 20,
    use_gpu: bool = True,
    top_k: List[int] = [3, 5],
) -> Dict[str, Any]:
    """
    Виконує інференс над одним CSV-файлом, обчислює метрики та будує графіки.

    Args:
        test_file: ім'я CSV-файлу в processed_dir.
        processed_dir: тека з обробленими даними.
        model_path: шлях до .h5 файлу моделі.
        window_size: розмір скользячого вікна.
        use_gpu: спробувати використовувати GPU.
        top_k: список k для top-k accuracy.

    Returns:
        Словник з ключами:
            timestamp, model_used, metrics (словар метрик),
            confusion_matrix_plot, roc_plot, pr_plot, calibration_plot,
            predictions, prediction_probs, true_labels, summary (HTML-таблиця),
            hardware ('GPU' або 'CPU').
    """
    logger.info("Інференс файлу %s з моделлю %s", test_file, model_path)

    # Налаштування апаратури
    is_gpu = setup_hardware(use_gpu)
    logger.info("Використовується %s", 'GPU' if is_gpu else 'CPU')

    # Завантаження моделі
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (UnknownError, OpError) as exc:
        logger.warning(
            "Помилка JIT на GPU (%s), перехід на CPU", type(exc).__name__
        )
        tf.keras.backend.clear_session()
        is_gpu = setup_hardware(False)
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(model_path, compile=False)

    # Підготовка даних
    df = pd.read_csv(os.path.join(processed_dir, test_file))
    X_raw, y_raw = prepare_data(df)
    num_classes = model.output_shape[-1]
    if int(y_raw.max()) >= num_classes:
        msg = "Мітка за межами кількості класів моделі"
        logger.error(msg)
        raise ValueError(msg)

    X_test, y_test_onehot = prepare_windowed_data(
        X_raw, y_raw, window_size, num_classes
    )
    y_true_labels = np.argmax(y_test_onehot, axis=1)

    # Інференс
    device = '/GPU:0' if is_gpu else '/CPU:0'
    try:
        with tf.device(device):
            inp = tf.convert_to_tensor(X_test, tf.float32)
            preds = model(inp, training=False).numpy()
    except Exception:
        logger.warning("Не вдалося на %s, перехід на CPU", device)
        tf.keras.backend.clear_session()
        tf.config.set_visible_devices([], 'GPU')
        is_gpu = setup_hardware(False)
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(model_path, compile=False)
            inp = tf.convert_to_tensor(X_test, tf.float32)
            preds = model(inp, training=False).numpy()

    y_pred_labels = np.argmax(preds, axis=1)

    # Обчислення метрик
    metrics: Dict[str, Any] = {
        'accuracy':    accuracy_score(y_true_labels, y_pred_labels),
        'precision':   precision_score(y_true_labels, y_pred_labels, average='macro'),
        'recall':      recall_score(y_true_labels, y_pred_labels, average='macro'),
        'f1':          f1_score(y_true_labels, y_pred_labels, average='macro'),
        'weighted_f1': f1_score(y_true_labels, y_pred_labels, average='weighted'),
        'log_loss':    log_loss(y_test_onehot, preds),
    }

    unique = np.unique(y_true_labels)
    for k in top_k:
        metrics[f'top_{k}_accuracy'] = (
            top_k_accuracy_score(
                y_true_labels, preds[:, unique], k=k, labels=unique
            ) if unique.size >= k else None
        )

    if unique.size >= 2:
        y_sub_true = y_test_onehot[:, unique]
        y_sub_pred = preds[:, unique]
        metrics.update({
            'roc_auc_micro': roc_auc_score(y_sub_true, y_sub_pred, average='micro'),
            'roc_auc_macro': roc_auc_score(y_sub_true, y_sub_pred, average='macro'),
            'pr_auc_micro':  average_precision_score(y_sub_true, y_sub_pred, average='micro'),
            'pr_auc_macro':  average_precision_score(y_sub_true, y_sub_pred, average='macro'),
        })
    else:
        metrics.update({
            'roc_auc_micro': None,
            'roc_auc_macro': None,
            'pr_auc_micro':  None,
            'pr_auc_macro':  None,
        })

    # Генерація графіків
    class_names = [str(i) for i in range(num_classes)]
    cm_plot    = plot_confusion_matrix(y_true_labels, y_pred_labels, class_names)
    roc_plot   = plot_roc_curves(y_test_onehot, preds, class_names)
    pr_plot    = plot_pr_curves(y_test_onehot, preds, class_names)
    calib_plot = plot_calibration(y_test_onehot, preds)

    # Підсумкова таблиця
    summary = pd.DataFrame({
        'Dataset': ['Test'],
        'Samples': [len(X_test)],
        'Classes': [num_classes],
    }).to_html(classes='table table-striped', index=False)

    return {
        'timestamp':             datetime.now().isoformat(),
        'model_used':            os.path.basename(model_path),
        'metrics':               metrics,
        'confusion_matrix_plot': cm_plot,
        'roc_plot':              roc_plot,
        'pr_plot':               pr_plot,
        'calibration_plot':      calib_plot,
        'predictions':           y_pred_labels.tolist(),
        'prediction_probs':      preds.tolist(),
        'true_labels':           y_true_labels.tolist(),
        'summary':               summary,
        'hardware':              'GPU' if is_gpu else 'CPU',
    }
