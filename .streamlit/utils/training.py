# app/utils/training.py
import os, io, base64, time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple
import multiprocessing
import json
import matplotlib

from tensorflow.python.framework.errors_impl import UnknownError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
from .logging_config import setup_logger

# Ініціалізація логера
logger = setup_logger('training', 'training')

matplotlib.use('Agg')
tf.config.optimizer.set_jit(False)

class TrainingProgressCallback(Callback):
    def __init__(self, progress_file):
        super().__init__()
        self.progress_file = progress_file
        self.current_epoch = 0
        self.total_epochs = 0
        self.final_metrics = None
        
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self._update_progress(0, "Початок тренування...")
        logger.info(f"Початок тренування на {self.total_epochs} епох")
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.progress_file and not os.path.exists(self.progress_file):
            logger.warning("Тренування зупинено користувачем")
            self.model.stop_training = True
            return

        self.current_epoch = epoch + 1
        percentage = int((epoch / self.total_epochs) * 100)
        self._update_progress(
            percentage,
            f"Епоха {self.current_epoch}/{self.total_epochs}"
        )
        logger.info(f"Початок епохи {self.current_epoch}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.progress_file and not os.path.exists(self.progress_file):
            logger.warning("Тренування зупинено користувачем")
            self.model.stop_training = True
            return

        metrics = {
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0))
        }
        self._update_progress(
            int((self.current_epoch / self.total_epochs) * 100),
            f"Епоха {self.current_epoch}/{self.total_epochs} - Втрати: {logs.get('loss', 0):.4f},"
             f" Точність: {logs.get('accuracy', 0):.4f}",
            metrics
        )
        logger.info(f"Завершено епоху {self.current_epoch}/{self.total_epochs} - "
                    f"Втрати: {logs.get('loss', 0):.4f}, Точність: {logs.get('accuracy', 0):.4f}")
        
    def on_train_end(self, logs=None):
        if self.final_metrics:
            self._update_progress(
                100,
                "Тренування успішно завершено!",
                self.final_metrics
            )
            logger.info("Тренування успішно завершено")
        
    def _update_progress(self, percentage, message, metrics=None):
        progress_data = {
            'percentage': percentage,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if metrics:
            progress_data['metrics'] = metrics
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)

class ProgressWithETA(Callback):
    def __init__(self, progress_file):
        super().__init__()
        self.progress_file = progress_file

    def on_train_begin(self, logs=None):
        self.total_epochs     = self.params['epochs']
        self.steps_per_epoch  = self.params['steps']
        self.total_batches    = self.total_epochs * self.steps_per_epoch
        self.train_start_time = time.time()


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
       # сколько батчей уже сделано с начала тренировки
        if self.progress_file and not os.path.exists(self.progress_file):
            self.model.stop_training = True
            return

        done = self.epoch * self.steps_per_epoch + (batch + 1)
        now  = time.time()
        elapsed = now - self.train_start_time

        # среднее время на батч
        avg_batch = elapsed / done if done else 0
        remaining_batches = self.total_batches - done
        eta_sec = int(avg_batch * remaining_batches)
        m, s = divmod(eta_sec, 60)
        eta_str = f"{m:02d}:{s:02d}"

        # глобальный процент
        percentage = int(done / self.total_batches * 100)

        data = {
            'percentage': percentage,
            'message': (
                f"Epoch {self.epoch+1}/{self.total_epochs}  "
                f"[{batch+1}/{self.steps_per_epoch}]  ETA: {eta_str}"
            ),
            'metrics': {
                'loss':        logs.get('loss'),
                'accuracy':    logs.get('accuracy'),
                'val_loss':    logs.get('val_loss'),
                'val_accuracy':logs.get('val_accuracy'),
            }
        }

        with open(self.progress_file, 'w') as f:
            json.dump(data, f)


def configure_gpu():
    """
    Конфігурує GPU лише якщо CUDA-збірка справжня і фізично є прилади
    """
    tf.config.optimizer.set_jit(False)
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.info("Фізичних GPU не знайдено, використовуємо CPU")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.keras.mixed_precision.set_global_policy('float32')
        logger.info(f"Використовуємо GPU: {gpus[0].name}")
        return True
    except RuntimeError as e:
        logger.error(f"Помилка конфігурації GPU: {e}")
        return False

def _initial_cpu_config():
    """
    Первична налаштування паралелізму CPU (одноразово, до першого імпорту/використання TF).
    """
    num_cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    logger.info(f"Первична налаштування CPU: {num_cores} ядер")

def configure_cpu(hide_gpu: bool = True, initial: bool = False):
    """
    Налаштування CPU:
      - initial=True  — тільки встановлення threading (викликати ДО першого створення моделей).
      - hide_gpu=True — приховує GPU-пристрої.
    """
    if hide_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU пристрої приховано")
        except Exception as e:
            logger.error(f"Не вдалося приховати GPU: {e}")
    if initial:
        try:
            _initial_cpu_config()
        except Exception as e:
            logger.error(f"Помилка первичної налаштування CPU: {e}")
    return True


def setup_hardware(use_gpu: bool = True) -> bool:
    """
    Configure hardware based on user preference
    """
    if not use_gpu:
        logger.info("Примусове використання CPU")
        configure_cpu()
        return False

    # user asked for GPU
    if configure_gpu():
        return True
    else:
        logger.info("GPU недоступний, перемикаємося на CPU")
        configure_cpu()
        return False

def prepare_windowed_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    window_size: int,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Розбиває послідовність на вікна фіксованого розміру
    """
    try:
        # Convert to numpy
        if hasattr(X, "values"):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if hasattr(y, "values"):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        y_arr = y_arr.squeeze()
        
        n_rows, n_feat = X_arr.shape
        n_windows = n_rows // window_size
        usable = n_windows * window_size
        
        if usable < n_rows:
            logger.info(f"Обрізано {n_rows - usable} рядків до {usable} ({n_windows} вікон)")
        
        X_trim = X_arr[:usable]
        y_trim = y_arr[:usable]
        
        # Reshape for CNN-LSTM
        X_windows = X_trim.reshape(n_windows, window_size, n_feat)
        y_last = y_trim.reshape(n_windows, window_size)[:, -1]
        
        # Convert to one-hot encoding
        y_windows = to_categorical(y_last, num_classes)
        
        # Ensure data types are correct
        X_windows = X_windows.astype(np.float32)
        y_windows = y_windows.astype(np.float32)
        
        logger.info(f"Підготовлено дані: {n_windows} вікон розміром {window_size}x{n_feat}")
        return X_windows, y_windows
    except Exception as e:
        logger.error(f"Помилка при підготовці віконних даних: {str(e)}")
        raise

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


def train_model_on_files(
    train_file: str,
    val_file: str,
    test_file: str,
    processed_dir: str,
    model_dir: str,
    batch_size: int = 64,
    epochs: int = 20,
    model_params: dict = None,
    use_gpu: bool = True,
    progress_file: str = None
) -> dict:
    """
    Train a CNN-LSTM model on CSV datasets with automatic GPU→CPU fallback.

    Args:
        train_file: filename of training CSV
        val_file: filename of validation CSV
        test_file: filename of test CSV
        processed_dir: directory path containing these files
        model_dir: where to save the trained model
        batch_size: initial batch size
        epochs: number of epochs
        model_params: custom architecture hyperparameters
        use_gpu: attempt to use GPU if True
        progress_file: path to JSON progress file

    Returns:
        dict of final metrics, plots (base64), model path, etc.
    """
    # --------------------
    # 1. Hardware setup
    # --------------------
    try:
        is_gpu = setup_hardware(use_gpu)
    except Exception as e:
        msg = f"⚠️ Ошибка при инициализации GPU: {e}. Переключаемся на CPU."
        print(msg)
        is_gpu = False

    if use_gpu and not is_gpu and progress_file:
        with open(progress_file, 'w') as f:
            json.dump({
                'percentage': 0,
                'message': "⚠️ GPU недоступний, переходимо на CPU і продовжуємо навчання...",
                'metrics': None,
                'timestamp': datetime.now().isoformat()
            }, f)

    # Adjust batch size
    batch_size = min(batch_size * 2, 256) if is_gpu else min(batch_size, 32)

    # --------------------
    # 2. Data loading/prep
    # --------------------
    train_df = pd.read_csv(os.path.join(processed_dir, train_file))
    val_df   = pd.read_csv(os.path.join(processed_dir, val_file))
    test_df  = pd.read_csv(os.path.join(processed_dir, test_file))

    X_train, y_train = prepare_data(train_df)
    X_val,   y_val   = prepare_data(val_df)
    X_test,  y_test  = prepare_data(test_df)

    WINDOW_SIZE = 20
    NUM_CLASSES = y_train.max() + 1

    X_train, y_train = prepare_windowed_data(X_train, y_train, WINDOW_SIZE, NUM_CLASSES)
    X_val,   y_val   = prepare_windowed_data(X_val,   y_val,   WINDOW_SIZE, NUM_CLASSES)
    X_test,  y_test  = prepare_windowed_data(X_test,  y_test,  WINDOW_SIZE, NUM_CLASSES)

    # --------------------
    # 3. Model & callbacks
    # --------------------
    default_params = {
        'filters': 64, 'kernel_size': 5, 'pool_size': 2,
        'lstm_units': 128, 'lstm_layers': 1,
        'dropout_rate': 0.3, 'recurrent_dropout': 0.1,
        'activation': 'relu', 'kernel_initializer': 'he_uniform',
        'optimizer': 'adam', 'learning_rate': 0.001
    }
    if model_params:
        default_params.update(model_params)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ProgressWithETA(progress_file)
    ]
    if progress_file:
        callbacks.append(TrainingProgressCallback(progress_file))

    # --------------------
    # 4. Training with fallback
    # --------------------
    try:
        # Attempt training on configured device
        model = create_model(
            input_shape=X_train.shape[1:],
            num_classes=y_train.shape[1],
            **default_params
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        if progress_file and not os.path.exists(progress_file):
            print("⚠️ Training stopped by user; aborting remaining steps.")
            return {}

    except UnknownError as e:
        # Fallback to CPU
        fallback_msg = (
            f"⚠️ Помилка JIT на GPU. Переключаемся на CPU і продовжуємо навчання..."
        )
        print(fallback_msg)
        if progress_file:
            with open(progress_file, 'w') as f:
                json.dump({
                    'percentage': 0,
                    'message': fallback_msg,
                    'metrics': None,
                    'timestamp': datetime.now().isoformat()
                }, f)
        with tf.device('/CPU:0'):
            model = create_model(
                input_shape=X_train.shape[1:],
                num_classes=y_train.shape[1],
                **default_params
            )
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        is_gpu = False

    # --------------------
    # 5. Evaluation & save
    # --------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    model.save(model_path)

    # Plots to base64
    loss_plot = plot_training_history(history.history['loss'], history.history['val_loss'], 'loss')
    acc_plot  = plot_training_history(history.history['accuracy'], history.history['val_accuracy'], 'accuracy')

    # Summary table
    summary = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Samples': [len(X_train), len(X_val), len(X_test)],
        'Classes': [NUM_CLASSES]*3
    }).to_html(classes='table table-striped', index=False)

    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'model_path': model_path,
        'loss_plot': loss_plot,
        'accuracy_plot': acc_plot,
        'summary': summary,
        'classes': [str(i) for i in range(NUM_CLASSES)],
        'history': history.history,
        'hardware': 'GPU' if is_gpu else 'CPU',
        'batch_size': batch_size
    }

    if progress_file:
        with open(progress_file, 'w') as f:
            json.dump({
                'percentage': 100,
                'message': 'Training completed successfully!',
                'metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }, f)

    return final_metrics


def create_model(
    input_shape: tuple,
    num_classes: int,
    filters: int = 64,
    kernel_size: int = 5,
    pool_size: int = 2,
    lstm_units: int = 128,
    lstm_layers: int = 1,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.1,
    activation: str = 'relu',
    kernel_initializer: str = 'he_uniform',
    optimizer: str = 'adam',
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Створення моделі з заданими параметрами
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # CNN layers
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding='same'  # Add padding to maintain sequence length
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
    
    # LSTM layers with modified configuration
    for i in range(lstm_layers):
        x = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=(i < lstm_layers - 1),
            dropout=dropout_rate,
            recurrent_dropout=0,  # Disable recurrent dropout to avoid JIT issues
            kernel_initializer=kernel_initializer,
            unroll=True  # Unroll short sequences to avoid JIT compilation
        )(x)
        if i < lstm_layers - 1:
            x = tf.keras.layers.BatchNormalization()(x)
    
    # Dropout
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=kernel_initializer
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with modified settings
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
        
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True,  # Force eager execution
        jit_compile=False  # Disable JIT compilation
    )
    
    return model

def plot_training_history(train_vals, val_vals, metric_name):
    fig, ax = plt.subplots()
    # рисуем линии
    ax.plot(train_vals, label='train')
    ax.plot(val_vals,   label='val')
    ax.set_title(f'{metric_name} over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def prepare_data(df):
    X = df.drop(columns=['label_code']).values
    y = df['label_code'].values
    return X, y