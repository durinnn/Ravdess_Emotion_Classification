import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking, BatchNormalization, Lambda
from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K

# 클래스와 벡터의 매핑
class_to_vector = {0: np.array([0, 0]), 1: np.array([0.3, -1]), 2: np.array([1, 0.3]), 
                   3: np.array([-1, -0.3]), 4: np.array([-0.7, 0.7]), 5: np.array([-0.3, 1]), 
                   6: np.array([-1, 0.3]), 7: np.array([0, 1])}

vector_list = tf.constant([
    [0, 0],    # neutral
    [0.3, -1], # calm
    [1, 0.3],  # happy
    [-1, -0.3],# sad
    [-0.7, 0.7], # angry
    [-0.3, 1], # fearful
    [-1, 0.3], # disgust
    [0, 1]     # surprise
], dtype=tf.float32)

# 거리 계산 함수 (정규화 없이)
def vector_to_class(vector):
    distances = tf.norm(vector_list - vector, axis=1)
    return tf.cast(tf.argmin(distances), tf.int32)


def cosine_similarity_loss(y_true, y_pred, epsilon=0.2):

    y_pred_magnitude = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1, keepdims=True))
    y_pred_is_zero = tf.cast(y_pred_magnitude < epsilon, tf.bool)
    
    y_pred_normalized = tf.math.l2_normalize(y_pred, axis=1)
    y_pred_normalized = tf.where(y_pred_is_zero, tf.zeros_like(y_pred_normalized), y_pred_normalized)
    
    cosine_similarity = tf.reduce_sum(y_true * y_pred_normalized, axis=1)
    loss = 1 - cosine_similarity
    
    return tf.reduce_mean(loss)

def custom_rmse(y_true, y_pred, epsilon=0.2):
    y_pred_magnitude = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1, keepdims=True))
    y_pred_is_zero = tf.cast(y_pred_magnitude < epsilon, tf.bool)
    
    y_pred_normalized = tf.math.l2_normalize(y_pred, axis=1)
    y_pred_normalized = tf.where(y_pred_is_zero, tf.zeros_like(y_pred_normalized), y_pred_normalized)
    
    error = y_true - y_pred_normalized
    squared_error = tf.square(error)
    mean_squared_error = tf.reduce_mean(squared_error, axis=1)
    rmse = tf.sqrt(mean_squared_error)
    
    return tf.reduce_mean(rmse)

# 커스텀 정확도 함수 정의. 예측벡터 클래스화 후, 분류정확도 측정
def custom_accuracy(y_true, y_pred):
    y_pred_magnitude = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1, keepdims=True))
    y_pred_is_zero = tf.cast(y_pred_magnitude < 0.2, tf.bool)
    
    y_pred_normalized = tf.math.l2_normalize(y_pred, axis=1)

    y_pred_normalized = tf.where(y_pred_is_zero, tf.zeros_like(y_pred_normalized), y_pred_normalized)
    y_true_classes = tf.map_fn(vector_to_class, y_true, fn_output_signature=tf.int32)
    y_pred_classes = tf.map_fn(vector_to_class, y_pred, fn_output_signature=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_classes, y_pred_classes), tf.float32))
    return accuracy


class CustomCallback(Callback):
    def __init__(self, validation_data, epsilon=0.2):
        super(CustomCallback, self).__init__()
        self.validation_data = validation_data
        self.epsilon = epsilon
        self.vector_list = tf.constant(list(class_to_vector.keys()), dtype=tf.float32)

    def on_epoch_end(self, epoch, logs=None):
        # 예측 함수와 예측 클래스 출력
        y_pred = self.model.predict(self.validation_data[0])
        
        # 예측 벡터 정규화
        y_pred = tf.constant(y_pred, dtype=tf.float32)
        y_pred_normalized = tf.math.l2_normalize(y_pred, axis=1)
        
        # 임계범위 내의 벡터를 (0, 0)으로 간주
        y_pred_magnitude = tf.sqrt(tf.reduce_sum(tf.square(y_pred_normalized), axis=1, keepdims=True))
        y_pred_is_zero = y_pred_magnitude < self.epsilon
        y_pred_normalized = tf.where(y_pred_is_zero, tf.zeros_like(y_pred_normalized), y_pred_normalized)
        
        # 진짜 벡터
        y_true = tf.constant(self.validation_data[1], dtype=tf.float32)
        
        y_pred_classes = tf.map_fn(lambda x: vector_to_class(x), y_pred_normalized, fn_output_signature=tf.int32)
        y_true_classes = tf.map_fn(lambda x: vector_to_class(x), y_true, fn_output_signature=tf.int32)
        
        print(f"Epoch {epoch+1}:")
        print(f"Predicted Vectors:\n{y_pred_normalized[:15]}")
        print(f"Predicted Classes:\n{y_pred_classes.numpy()[:15]}")
        print(f"True Classes:\n{y_true_classes.numpy()[:15]}")


def create_lstm_model():

    model = Sequential()
    model.add(Masking(mask_value=0.))
    model.add(LSTM(128, dropout=0.1, return_sequences=True))
    model.add(LSTM(128, dropout=0.1, return_sequences=False)) 
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))  # 2차원 벡터를 예측

    model.compile(optimizer='adam', loss=custom_rmse, metrics=[custom_accuracy])
    return model



def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=32):

    # 조기 종료 콜백 설정
    custom_callback = CustomCallback(validation_data=(X_test, y_test))
    earlystop = EarlyStopping(monitor='val_custom_accuracy', mode='max', patience=100, restore_best_weights=True)
    # 모델 학습
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[earlystop])

    # 모델 평가
    predictions = model.predict(X_test)
    predicted_classes = [vector_to_class(pred) for pred in predictions]
    true_classes = [vector_to_class(true_vector) for true_vector in y_test]

    return history, predicted_classes, true_classes
