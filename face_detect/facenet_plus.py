import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os
import time

# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 모델 초기화
mtcnn = MTCNN(keep_all=False, device=device)  # 얼굴 검출용
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # 얼굴 임베딩 추출용

# 얼굴 비교 함수 (유클리드 거리 계산)
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def extract_embedding_and_boxes(image):
    """이미지에서 얼굴 검출 후 첫 번째 얼굴의 임베딩과 바운딩 박스를 반환"""
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:  # 얼굴이 하나 이상 검출된 경우
        # 첫 번째 얼굴만 처리
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            print("Empty face region detected.")
            return None, None
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
        return embedding, [boxes[0]]  # 첫 번째 바운딩 박스만 반환
    else:
        print("No face detected.")
        return None, None

def save_new_face(image, folder_path):
    """새로운 얼굴 이미지를 저장"""
    file_name = input("이름을 입력하세요: ") + ".jpg"  # 사용자로부터 이름 입력
    cv2.imwrite(os.path.join(folder_path, file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"새로운 얼굴이 저장되었습니다: {file_name}")

# 폴더 내 모든 이미지 로드 및 임베딩 추출
def load_embeddings_from_folder(folder_path):
    embeddings = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image from {file_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding, boxes = extract_embedding_and_boxes(img_rgb)
        if embedding is not None:
            embeddings[file_name] = embedding
    return embeddings

# 기준 이미지 폴더 설정
img_src_folder = 'img_src'
reference_embeddings = load_embeddings_from_folder(img_src_folder)
if not reference_embeddings:
    print("No valid face embeddings found in the folder.")
    exit()
else:
    print(f"Loaded {len(reference_embeddings)} reference embeddings.")

# 웹캠으로 실시간 얼굴 검출 및 비교
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("Press 'q' to quit.")
match_start_time = None  # 매칭된 시간을 저장
no_match_start_time = None  # 매칭되지 않은 시간을 기록
matched = False  # 매칭 상태를 저장

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break
    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 실시간 얼굴 임베딩 추출
    embedding, boxes = extract_embedding_and_boxes(frame_rgb)

    if embedding is not None:
        best_match = "No Match"
        min_distance = float('inf')

        # 기준 얼굴들과 비교
        for file_name, ref_embedding in reference_embeddings.items():
            distance = calculate_distance(ref_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = file_name if min_distance < 0.7 else "No Match"

        if best_match != "No Match":
            if not matched:  # 처음 매칭된 경우만 시간 기록
                print(f"{best_match}님 환영합니다. 3초 뒤 종료됩니다.")
                match_start_time = time.time()
                matched = True
                no_match_start_time = None  # 매칭되면 비매칭 시간 초기화

            # 바운딩 박스와 이름 표시
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 3초 후 종료
            if matched and time.time() - match_start_time > 3:
                break
        else:
            matched = False  # 매칭이 실패한 경우 상태 초기화
            if no_match_start_time is None:  # 비매칭 시작 시간 기록
                no_match_start_time = time.time()

            # 매칭되지 않고 2초 이상 경과한 경우
            if no_match_start_time and time.time() - no_match_start_time > 2:
                print("새로운 얼굴이 감지되었습니다. 이미지를 저장합니다.")
                save_new_face(frame_rgb, img_src_folder)
                reference_embeddings = load_embeddings_from_folder(img_src_folder)  # 새로 로드
                no_match_start_time = None  # 비매칭 시간 초기화

            # 매칭되지 않은 경우 바운딩 박스 표시
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "No Match", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        print("No face detected.")

    # 화면 출력
    cv2.imshow("Webcam Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
