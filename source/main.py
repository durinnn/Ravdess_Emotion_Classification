#main
import numpy as np
from sklearn.model_selection import KFold
import lstm, lstm_vector
import librosa
import librosa.display
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle

def plot_mel(src):  
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(np.transpose(src), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
def plot_mfcc(mfccs):

    librosa.display.specshow(np.transpose(mfccs), x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_model_history(history, filename):

    filename += '_graph'

    # Summarize history for accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_model_history_custom(history, filename):

    filename += '_graph'

    # Summarize history for custom accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['custom_accuracy'])
    plt.plot(history.history['val_custom_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(filename)
    plt.close()

def evaluate_model(predicted_classes, true_classes, filename):

    # 라벨 딕셔너리
    label_dict = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprise'}

    # 혼동 행렬 계산
    cm = confusion_matrix(predicted_classes, true_classes)

    # 혼동 행렬 출력
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename + '_chart')
    plt.close()    

    # Classification report 출력
    report = classification_report(true_classes, predicted_classes, target_names=label_dict.values())
    with open(filename + '_report', 'w') as f:
        f.write(report)



def kfold(data):

    num_classes = 8

    X = [item[0] for item in data] #각 데이터 순서따라 mel/mfcc만 한 배열에 저장
    y = [item[1] for item in data]

    kf = KFold(n_splits=5, shuffle=True, random_state=312)
    # k-폴드 교차 검증
    for fold, (train_index, test_index) in enumerate(kf.split(data)):

        #데이터 numpy로 전처리
        X_train = np.array([X[i] for i in train_index])
        y_train = np.array([y[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_test = np.array([y[i] for i in test_index])
        # 원핫 인코딩
        y_train_onehot = to_categorical(y_train, num_classes)
        y_test_onehot = to_categorical(y_test, num_classes)
        # 모델 입력
        # X.shape : (num_samples, num_timesteps, num_features) 시간 축의 길이, 각 시간스텝 별 특징 수(필터수)
        # 모델 학습 및 평가


        model = lstm.create_lstm_model(num_classes)

        history, predicted_classes, true_classes = lstm.train_and_evaluate_model(model, X_train, y_train_onehot, X_test, y_test_onehot)
        filename = str(fold) + '_mfcc_origin'

        #결과파일 생성
        plot_model_history(history, filename)
        evaluate_model(predicted_classes, true_classes, filename)

def kfold_vector(data):

    X = np.array([item[0] for item in data])  # 각 데이터 순서대로 feature를 배열에 저장
    y = np.array([item[2] for item in data])  # 각 데이터 순서대로 레이블을 배열에 저장

    kf = KFold(n_splits=5, shuffle=True, random_state=31)

    # k-폴드 교차 검증
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        #데이터 numpy로 전처리
        X_train = np.array([X[i] for i in train_index])
        y_train = np.array([y[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_test = np.array([y[i] for i in test_index])

        # 모델 입력
        # X.shape : (num_samples, num_timesteps, num_features) 시간 축의 길이, 각 시간스텝 별 특징 수(필터수)
        # 레이블 y 형태는 2차원 벡터

        # 데이터 정규화 (패딩 부분 유지)

       # 모델 생성 및 요약
        model = lstm_vector.create_lstm_model()

        history, predicted_classes, true_classes = lstm_vector.train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        filename = str(fold) + '_mel40_rmse_minmax_vector'

        #결과파일 생성
        plot_model_history_custom(history, filename)
        evaluate_model(predicted_classes, true_classes, filename)
        break


with open('melspecdata.pkl', 'rb') as f:
    melspecdata = pickle.load(f)

with open('mfccdata.pkl', 'rb') as f:
    mfccdata = pickle.load(f)

with open('melspecdata_n.pkl', 'rb') as f:
    melspecdata_n = pickle.load(f)

with open('mfccdata_n.pkl', 'rb') as f:
    mfccdata_n = pickle.load(f)


kfold(melspecdata)
kfold(mfccdata)
