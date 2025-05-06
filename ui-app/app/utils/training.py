# app/utils/training.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from app.models.architecture import create_model


def train_model_on_csv(
    filename: str,
    processed_dir: str,
    model_dir: str,
    batch_size: int = 64,
    epochs: int = 20,
    test_size: float = 0.15,
    val_split: float = 0.12
) -> dict:
    """
    1) Завантажує оброблений CSV;
    2) Формує X, y; кодує мітки в [0..n_classes-1];
    3) Розбиває на train/val/test із стратифікацією;
    4) One-hot-енкодинг y для Keras;
    5) Ініціалізує модель через create_model();
    6) Навчає з EarlyStopping та ReduceLROnPlateau;
    7) Оцінює на тесті;
    8) Зберігає ваги в HDF5;
    9) Повертає словник із history + фінальними метриками.
    """
    # шляхи
    csv_path   = os.path.join(processed_dir, filename)
    model_path = os.path.join(model_dir, f"{os.path.splitext(filename)[0]}_model.h5")

    # 1) load data
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Label']).values
    raw_y = df['Label'].values

    # 2) encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(raw_y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_int, num_classes)

    # 3) split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat,
        test_size=test_size,
        stratify=y_int,
        random_state=42
    )
    val_relative = val_split / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=np.argmax(y_temp, axis=1),
        random_state=42
    )

    # 4) create model
    num_features = X.shape[1]
    model = create_model(
        window_size=None,           # у create_model дозволяємо None → просто Dense
        num_features=num_features,
        num_classes=num_classes
    )

    # 5) callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # 6) fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # 7) evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # 8) save
    model.save(model_path)

    # 9) збираємо результати
    result = {
        'model_path': model_path,
        'history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
        },
        'test_loss': round(float(loss), 4),
        'test_accuracy': round(float(acc), 4),
        'classes': list(le.classes_)
    }
    return result