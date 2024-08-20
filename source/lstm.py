#lstm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np



        
def create_lstm_model(num_classes):  # 파라미터 : 출력 클래스 (분류할 카테고리)
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, 13)))  # Masking층 추가. 값0 패딩 무시, 인풋 데이터 형태 지정
    model.add(LSTM(128, dropout=0.1, return_sequences=True))  # 128유닛 LSTM레이어
    model.add(LSTM(128, dropout=0.1, return_sequences=False))  # 128노드 LSTM레이어
    model.add(BatchNormalization())  # 마지막 파라미터 feature를 정규화
    model.add(Dense(num_classes, activation='softmax'))  # n개의 출력층 가지게, 활성화 함수로 softmax 사용
    
    # rmsprop 최적화 알고리즘 사용, loss 함수로 categorical crossentropy, 평가지표로 정확도를 사용
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#1440개 중, kfold하여 분리한 데이터를 한회독에 32배치씩, 36번 훈련(1 에폭). 이를 500번 수행
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=32):

    earlystop = EarlyStopping(monitor='val_accuracy', mode='max', patience=50, restore_best_weights=True)

    # 모델 학습
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[earlystop])

    # 모델 평가
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    return history, predicted_classes, true_classes