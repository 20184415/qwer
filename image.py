from flask import Flask, jsonify, send_file, Response
import torch
import cv2
import numpy as np
import os

app = Flask(__name__)

# YOLOv5 모델 로드
model_path = r'C:\Users\ych61\Downloads\b.pt'  # 학습된 모델 게주치 경로
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 빈과의 영역 검진을 위한 HSV 범위 설정
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])


# 익음 정도를 분류하는 함수
def classify(red):
    if red >= 90:
        return "완숙기"
    elif 61 <= red < 90:
        return "담적색기"
    elif 31 <= red < 60:
        return "도색기"
    elif 11 <= red < 30:
        return "재색기"
    elif 3 <= red < 10:
        return "변색기"
    else:
        return "녹숙기"


@app.route('/process_image', methods=['GET'])
def process_image():
    image_path = r'C:\Users\ych61\OneDrive\바탕 화면\tomato.png'




    img = cv2.imread(image_path)
    #img = cv2.resize(img, (1080, 1920))
    if img is None:
        return jsonify({"error": "이미지를 읽는 데 실패했습니다."})
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # YOLOv5 모델로 토마토 감지
    results = model(img)
    boxes = results.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표

    ripeness_info = []

    # 각 바운딩 박스 내에서 빨간색 비율 계산
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])  # 바운딩 박스 좌표
        tomato_region = hsv_img[y1:y2, x1:x2]  # 바운딩 박스 내 영역 추출

        # 빨간색 마스크 생성
        red_mask1 = cv2.inRange(tomato_region, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(tomato_region, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2

        # 바운딩 박스 내 전체 픽셀 수와 빨간색 픽셀 수 계산
        total_pixels = tomato_region.shape[0] * tomato_region.shape[1]
        red_pixels = np.sum(red_mask > 0)

        # 익은 비율 계산
        if total_pixels > 0:
            ripeness_ratio = (red_pixels / total_pixels) * 100
        else:
            ripeness_ratio = 0

        # 익음 정도 분류
        ripeness_stage = classify(ripeness_ratio)

        # 바운딩 박스와 익음 정도 텍스트 추가
        color = (0, 0, 255)  # 빨간색
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{ripeness_stage} ({ripeness_ratio:.2f}%)'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 결과 저장
        ripeness_info.append({
            'ripeness_ratio': ripeness_ratio,  # 빨간색 비율(익은 정도)
            'ripeness_stage': ripeness_stage  # 익음 단계
        })

    # 바운딩 박스와 익음 정도가 표시된 이미지 저장
    output_path = r'C:\Users\ych61\OneDrive\바탕 화면\tomato_processedq.png'
    success = cv2.imwrite(output_path, img)
    if not success:
        return jsonify({"error": "이미지를 저장하는 데 실패했습니다."})

    response = {
        "ripeness_info": ripeness_info,
    }

    return jsonify(response)


@app.route('/download_image', methods=['GET'])
def download_image():
    output_path = r'C:\Users\ych61\OneDrive\바탕 화면\tomato_processed.png'
    if os.path.exists(output_path):
        return send_file(output_path, mimetype='image/png', as_attachment=True)
    else:
        return jsonify({"error": "이미지가 존재하지 않습니다."})


if __name__ == '__main__':
    app.run(debug=True)