#loaddata
import os

# RAVDESS 데이터셋의 루트 디렉토리 경로
root_dir = "resource\\revdess"

# 감정 종류를 나타내는 사전. 각각 감정을 감정원형모형의 좌표상 정도에 따라 표현.
emotion_dict = {
    "01": (0, 0), #neutral
    "02": (0.3, -1.), #calm
    "03": (1., 0.3), #happy
    "04": (-1, -0.3), #sad
    "05": (-0.7, 0.7), #angry
    "06": (-0.3, 1.), #fearful
    "07": (-1., 0.3), #disgust
    "08": (0, 1.) #surprise
}

def getdata():
    # 데이터를 저장할 리스트
    data = []

    # 각 배우의 폴더를 순회
    for actor in range(1, 25):
        actor_folder = f"Actor_{actor:02d}"
        actor_path = os.path.join(root_dir, actor_folder)
        
        # 각 배우 폴더 내의 파일을 순회
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):  # wav 파일만 처리
                # 파일 경로
                file_path = os.path.join(actor_path, filename)
                
                # 파일명에서 정보를 추출
                parts = filename.split("-")
                emotion_code = parts[2]
                intensity_code = parts[3]
                
                # 감정 종류와 강도를 사전에서 찾기
                emotion = emotion_dict[emotion_code]
                intensity = 2 if intensity_code == "02" else 1            #normal = 1, strong = 2
                
                # 리스트에 추가
                data.append((file_path, emotion, intensity))
    
    return data