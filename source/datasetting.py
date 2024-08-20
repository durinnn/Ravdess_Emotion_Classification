#datasetting
import librosa
import librosa.display
import numpy as np
import loaddata
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

label_dict = {
    (0, 0) : 0, #neutral
    (0.3, -1) : 1, #calm
    (1, 0.3) : 2, #happy
    (-1, -0.3) : 3, #sad
    (-0.7, 0.7) : 4, #angry
    (-0.3, 1): 5, #fearful
    (-1, 0.3): 6, #disgust
    (0, 1): 7 #surprise
}


#모든 음성데이터들을 리스트에 주소, 감정정보와, 레벨과 함께 저장
alldata = loaddata.getdata()


def testp(src):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(src, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def testmf(src):

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(src, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.show()


def filter_mel_spectrogram(mel_spec, sr, n_fft, fmin=50, fmax=2000):
    # Mel 필터 생성
    mel_filter = librosa.filters.mel(n_mels=40, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax)
    mel_spec_filtered = np.dot(mel_filter.T, mel_spec)
    return mel_spec_filtered

#각각 다른 음성데이터의 시작 임계값으로 검사
def search_point(mel_spec, sr, n_fft, threshold = 0.00015):  # 각 시점의 에너지를 탐색

    mel_spec_filtered = filter_mel_spectrogram(mel_spec, sr, n_fft)   # 음성주파수영역에서의 임계값이상의 에너지일때, 음성 시작으로 인식
    
    # RMS 에너지 계산
    rms_energy = np.sqrt(np.mean(mel_spec_filtered, axis=0))

    start = None
    end = None

    # 음성 시작 지점 찾기
    for i, e in enumerate(rms_energy):
        if e > threshold:
            start = int(i)
            break
  
 

    # 음성 끝 지점 찾기 (역순 탐색)
    threshold = 0.00005
    for i in range(len(rms_energy) - 1, -1, -1):
        if rms_energy[i] > threshold:
            end = int(i) + 1
            break

    return start, end

def wav_to_melspectrogram(path, preemphasis_coeff=0.95, n_mels=40, hop_length=512):
    # WAV 파일 로드
    y, sr = librosa.load(path, sr=None)  # wav 파일 로드, 샘플링 레이트는 원본 유지

    # Preemphasis 적용
    y = np.append(y[0], y[1:] - preemphasis_coeff * y[:-1])

    n_fft = int(sr * 0.02)  # 윈도우 크기, 20ms
    hop_length = int(sr * 0.01)  # 이동 크기, 10ms
    
    # STFT 계산. 기본으로 leakage 방지를 위한 Hann window가 적용
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Mel spectrogram 계산. S는 STFT 결과의 절대값 제곱을 입력으로. n_mels는 필터 수
    mel_spec = librosa.feature.melspectrogram(S=np.abs(D)**2, sr=sr, n_mels=n_mels, power=2.0)

    #임계점을 넘는 주파수부터 시작으로 하여, 이전에 녹음된 다른소리는 제외
    start_idx, end_idx = search_point(mel_spec, sr, n_fft)
    # Mel spectrogram 자르기
    mel_spec = mel_spec[:, start_idx:end_idx]

    # log scale 정규화
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec

# alldata에서, 각 파일의 주소로 접근하여, wav 파일들을 Mel spectrogram으로 변환
def process_wav_files(data):
    mel_specs = []
    for one in data: 
        mel_spec = [wav_to_melspectrogram(one[0]), label_dict[one[1]], one[1], one[2]]    #mel_spec 안에, 각 파일의 mel_spectrogram, 감정레이블, 감정벡터, 감정레벨이 저장.
        mel_specs.append(mel_spec)

    return mel_specs

# Mel Spectrogram을 기반으로 MFCC 계산
def calculate_mfccs(mel_specs, n_mfcc=13):
    mfccs = []
    for mel_spec in mel_specs:
        # librosa.feature.mfcc() 함수를 사용하여 MFCC 계산. (feature, timestep)
        mfcc = [librosa.feature.mfcc(S=mel_spec[0], n_mfcc=n_mfcc), mel_spec[1], mel_spec[2], mel_spec[3]]
        mfccs.append(mfcc)
    return mfccs

def padding(data):
    padded_data = []

    lengths = [item[0].shape[1] for item in data]
    max_len = max(lengths)
    padded_item = [np.pad(item[0], ((0, 0), (0, max_len-item[0].shape[1])), mode='constant') for item in data]

    for i, item in enumerate(data):
        new_item = (np.transpose(padded_item[i]), item[1], item[2], item[3])
        padded_data.append(new_item)

    return padded_data

#각 오브젝트 안에, 각 파일의 feature, 감정레이블, 감정벡터, 감정레벨이 저장.
def transdata():

    # Mel spectrogram으로 변환된 파일들.
    mel_specs = process_wav_files(alldata) 

    #길이를 일정하게 하기 위해, padding 과정만 추가.
    padded_mel_specs = padding(mel_specs)

    # 앞서 계산된 Mel spectrogram과 샘플링 레이트(sr)를 사용하여 MFCC 계산
    mfccs = calculate_mfccs(mel_specs)
    
    padded_mfccs = padding(mfccs)

    #정규화 후 패딩하는 버전 추가
    scaler = MinMaxScaler()
    for mel in mel_specs:
        data_scaled = scaler.fit_transform(mel[0])
        mel[0] = data_scaled
    
    padded_mel_specs_normalized = padding(mel_specs)

    for mfcc in mfccs:
        data_scaled = scaler.fit_transform(mfcc[0])
        mfcc[0] = data_scaled
    
    padded_mfcc_normalized = padding(mfccs)

    return padded_mel_specs, padded_mfccs, padded_mel_specs_normalized, padded_mfcc_normalized

melspecdata, mfccdata, melspecdata_n, mfccdata_n = transdata() #각 오브젝트 안에, 각 파일의 feature, 감정레이블, 감정벡터, 감정레벨이 저장.
# 둘 다 (timestep, feature)

# 데이터를 파일로 저장
with open('melspecdata.pkl', 'wb') as f:
    pickle.dump(melspecdata, f)

with open('mfccdata.pkl', 'wb') as f:
    pickle.dump(mfccdata, f)

with open('melspecdata_n.pkl', 'wb') as f:
    pickle.dump(melspecdata_n, f)

with open('mfccdata_n.pkl', 'wb') as f:
    pickle.dump(mfccdata_n, f)