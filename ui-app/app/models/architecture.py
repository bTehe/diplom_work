import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU,
    MaxPooling1D, Dropout, Bidirectional, LSTM,
    Dense, Multiply, Softmax, Lambda
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def create_model(
    window_size: int,
    num_features: int,
    num_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    pool_size: int = 2,
    lstm_units: int = 128,
    lstm_layers: int = 1,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.1,
    activation: str = 'relu',
    kernel_initializer: str = 'he_uniform',
    optimizer='adam',
    optimizer__learning_rate: float = 1e-3
) -> Model:
    """
    Создаёт компилированную keras-модель со следующей архитектурой:
      – 1D-CNN блок
      – BiLSTM блок
      – Механизм внимания
      – Классификационная голова

    Параметры:
      window_size          – размер временного окна (число временных шагов)
      num_features         – число признаков на каждый шаг
      num_classes          – число выходных классов
      filters, kernel_size, pool_size, ... – гиперпараметры модели
      optimizer            – либо строка (имя оптимизатора), либо класс оптимизатора Keras,
                              либо уже инстанцированный оптимизатор
      optimizer__learning_rate – параметр learning_rate для оптимизатора, если это класс
    """
    # Вход
    inp = Input(shape=(window_size, num_features))
    x = inp

    # 1D-CNN блок
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters, kernel_size + 2, padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPooling1D(pool_size)(x)
    x = Dropout(dropout_rate)(x)

    # BiLSTM блок
    for _ in range(lstm_layers):
        x = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout
            )
        )(x)
    x = Dropout(dropout_rate)(x)

    # Механизм внимания
    attn_scores = Dense(1, activation='tanh')(x)
    attn_scores = Softmax(axis=1)(attn_scores)
    context = Multiply()([x, attn_scores])
    context = Lambda(lambda z: K.sum(z, axis=1))(context)

    # Классификационная голова
    x = Dense(128, activation=activation,
              kernel_initializer=kernel_initializer)(context)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    # Подбираем оптимизатор
    if isinstance(optimizer, type):
        opt = optimizer(learning_rate=optimizer__learning_rate)
    else:
        # строка или уже инстанцированный оптимизатор
        opt = optimizer

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
