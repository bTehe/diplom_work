import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU,
    MaxPooling1D, Dropout, Bidirectional, LSTM,
    Dense, Multiply, Softmax, Lambda
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

import kerastuner as kt
from kerastuner import HyperModel

# 1) Налаштування розподіленої стратегії на GPU
strategy = tf.distribute.MirroredStrategy()
print(f'Кількість реплік (GPU): {strategy.num_replicas_in_sync}')

# 2) Завантаження даних
def load_and_trim(path_X, path_y, drop_cols):
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y)
    to_drop_X = [X.columns[0]] + drop_cols
    to_drop_y = [y.columns[0]]
    X = X.drop(to_drop_X, axis=1)
    y = y.drop(to_drop_y, axis=1)
    return X, y


drop_cols = ["Init_Win_bytes_forward"]

X_train, y_train = load_and_trim(
    "NN Datasets/x_train.csv",
    "NN Datasets/y_train.csv",
    drop_cols
)

X_val, y_val = load_and_trim(
    "NN Datasets/x_val.csv",
    "NN Datasets/y_val.csv",
    drop_cols
)

X_test, y_test = load_and_trim(
    "NN Datasets/x_test.csv",
    "NN Datasets/y_test.csv",
    drop_cols
)


# 3) Параметри
window_size  = 20
num_features = X_train.shape[1]
class_mapping = {
    'BENIGN': 0, 'DoS': 1, 'PortScan': 2,
    'Bot_Infiltration': 3, 'Web': 4,
    'FTP_SSH_Patator': 5, 'Heartbleed': 6
}
num_classes = len(class_mapping)
class_names = [None]*num_classes
for name, idx in class_mapping.items():
    class_names[idx] = name

# 4) Підготовка віконних даних
def prepare_windowed_data(X, y, window_size, num_classes):
    X_arr = X.values
    y_arr = y.values.squeeze()
    n_rows, n_feat = X_arr.shape
    n_windows = n_rows // window_size
    n_use = n_windows * window_size
    if n_use != n_rows:
        print(f"– Обрізаю {n_rows - n_use} рядків")
    X_trim = X_arr[:n_use].reshape(n_windows, window_size, n_feat)
    y_trim = y_arr[:n_use].reshape(n_windows, window_size)[:,-1]
    y_cat  = to_categorical(y_trim, num_classes)
    return X_trim, y_cat

X_train, y_train = prepare_windowed_data(X_train, y_train, window_size, num_classes)
X_val,   y_val   = prepare_windowed_data(X_val,   y_val,   window_size, num_classes)
X_test,  y_test  = prepare_windowed_data(X_test,  y_test,  window_size, num_classes)

# 5) Колбеки
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# 6) Опис гіпер-моделі з підтримкою batch_size та epochs
class WindowedHyperModel(HyperModel):
    def __init__(self, window_size, num_features, num_classes, strategy, callbacks):
        self.window_size  = window_size
        self.num_features = num_features
        self.num_classes  = num_classes
        self.strategy     = strategy
        self.callbacks    = callbacks

    def build(self, hp):
        with self.strategy.scope():





            
            inp = Input(shape=(self.window_size, self.num_features))
            x = inp

            # CNN блок
            filters     = hp.Choice('filters',      [32, 64, 128])
            kernel_size = hp.Choice('kernel_size',  [3, 5, 7])
            pool_size   = hp.Choice('pool_size',    [2, 3])
            dropout_rate= hp.Choice('dropout_rate', [0.2, 0.3])
            rec_dropout = hp.Choice('recurrent_dropout', [0.1, 0.2])

            x = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x); x = ReLU()(x)
            x = Conv1D(filters, kernel_size+2, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x); x = ReLU()(x)
            x = MaxPooling1D(pool_size)(x)
            x = Dropout(dropout_rate)(x)

            # BiLSTM блок
            lstm_units  = hp.Choice('lstm_units',  [64, 128, 256])
            lstm_layers = hp.Choice('lstm_layers', [1, 2])
            for _ in range(lstm_layers):
                x = Bidirectional(LSTM(lstm_units,
                                       return_sequences=True,
                                       dropout=dropout_rate,
                                       recurrent_dropout=rec_dropout))(x)
            x = Dropout(dropout_rate)(x)

            # Attention
            attn = Dense(1, activation='tanh')(x)
            attn = Softmax(axis=1)(attn)
            ctx  = Multiply()([x, attn])
            ctx  = Lambda(lambda z: K.sum(z, axis=1))(ctx)

            # Голова класифікації
            activation = 'relu'
            x = Dense(128, activation=activation, kernel_initializer='he_uniform')(ctx)
            x = Dropout(dropout_rate)(x)
            out = Dense(self.num_classes, activation='softmax')(x)

            model = Model(inp, out)

            # Оптимізатор
            lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    # Переопреділяємо fit, щоб підхопити batch_size та epochs
    def fit(self, hp, model, x, y, validation_data, **kwargs):
        batch_size = hp.Choice('batch_size', [128, 256, 1024])
        epochs     = hp.Choice('epochs',     [10, 20, 50])
        return model.fit(
            x, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            **kwargs
        )

BASE_DIR = os.path.abspath('kt_logs')
PROJECT_NAME = 'bayes_opt_new'  # новий project_name

if tf.io.gfile.exists(BASE_DIR):
    tf.io.gfile.rmtree(BASE_DIR)

tf.io.gfile.makedirs(BASE_DIR)

print("Working dir:", os.getcwd())
print("BASE_DIR exists?", os.path.exists(BASE_DIR))
print("  isdir?", os.path.isdir(BASE_DIR))
print("  isfile?", os.path.isfile(BASE_DIR))
print("Absolute path:", os.path.abspath(BASE_DIR))
print("Contents of parent:", os.listdir(os.path.dirname(os.path.abspath(BASE_DIR)) or "."))

# 7) Налаштування та запуск Bayesian Optimization
hypermodel = WindowedHyperModel(window_size, num_features, num_classes, strategy, callbacks)

tuner = kt.BayesianOptimization(
    hypermodel,
    objective='val_loss',
    max_trials=30,               # скільки різних комбінацій спробувати
    directory=BASE_DIR,
    project_name=PROJECT_NAME,
    overwrite=True
)

print(f"KerasTuner пише логи у: {os.path.abspath(BASE_DIR)}/{PROJECT_NAME}/")


tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val)
)

# 8) Видобування найкращих гіперпараметрів і моделі
best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("Найкращі гіперпараметри:")
for param in best_hp.values.keys():
    print(f"  {param}: {best_hp.get(param)}")

# 9) Оцінка на тестовому наборі
y_prob = best_model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

test_acc      = np.mean(y_pred == y_true)
test_bal_acc  = balanced_accuracy_score(y_true, y_pred)
test_mcc      = matthews_corrcoef   (y_true, y_pred)
test_roc_auc  = roc_auc_score       (y_test, y_prob,
                                     multi_class='ovo', average='macro')
test_avg_prec = average_precision_score(y_test, y_prob,
                                        average='macro')

print("\nМетрики на тесті:")
print(f"Accuracy:              {test_acc:.4f}")
print(f"Balanced Accuracy:     {test_bal_acc:.4f}")
print(f"MCC:                   {test_mcc:.4f}")
print(f"ROC AUC (ovo, macro):  {test_roc_auc:.4f}")
print(f"Average Precision:     {test_avg_prec:.4f}")

# 10) (За бажанням) Збереження результатів
results = {
    'test_accuracy':              test_acc,
    'test_balanced_accuracy':     test_bal_acc,
    'test_mcc':                   test_mcc,
    'test_roc_auc':               test_roc_auc,
    'test_average_precision':     test_avg_prec,
}
pd.DataFrame([results]).to_csv('bayes_opt_test_metrics.csv', index=False)
