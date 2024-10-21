import os
import torch
import cv2
import numpy as np

# YOLOv5 모델 로드 (C:\Users\ych61\Downloads\best-tomato.pt 경로의 가중치 사용)
model_path = r'C:\Users\ych61\Downloads\best-tomato.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# HSV 범위 설정
red_lower1 = np.array([0, 100, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 50])
red_upper2 = np.array([180, 255, 255])
orange_lower = np.array([11, 100, 100])
orange_upper = np.array([25, 255, 255])
yellow_lower = np.array([26, 100, 100])
yellow_upper = np.array([34, 255, 255])
green_lower = np.array([35, 100, 100])
green_upper = np.array([85, 255, 255])

def process_image(image_path):

    results = model(image_path)
    boxes = results.xyxy[0].cpu().numpy()  # 바운딩 박스

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 이미지 파일을 읽을 수 없습니다. 경로를 확인하세요: {image_path}")
        return []
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ripeness_info = []

    # 각 바운딩 박스 내에서 색상 비율 계산
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # 바운딩 박스 크기 조정 (중앙 중심부로 축소)
        x1 = x1 + int((x2 - x1) * 0.15)  # 좌
        x2 = x2 - int((x2 - x1) * 0.15)  # 우
        y1 = y1 + int((y2 - y1) * 0.1)  # 상단
        y2 = y2 - int((y2 - y1) * 0.1)  # 하단
        box_region = hsv_img[y1:y2, x1:x2]

        # 색상 마스크 생성 및 비율 계산
        total_pixels = box_region.shape[0] * box_region.shape[1]
        if total_pixels == 0:  # 잘못된 박스는 무시
            continue
        red_mask = cv2.inRange(box_region, red_lower1, red_upper1) + cv2.inRange(box_region, red_lower2, red_upper2)
        orange_mask = cv2.inRange(box_region, orange_lower, orange_upper)
        yellow_mask = cv2.inRange(box_region, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(box_region, green_lower, green_upper)

        red_ratio = (np.sum(red_mask > 0) / total_pixels) * 100
        orange_ratio = (np.sum(orange_mask > 0) / total_pixels) * 100
        yellow_ratio = (np.sum(yellow_mask > 0) / total_pixels) * 100
        green_ratio = (np.sum(green_mask > 0) / total_pixels) * 100

        # 결과 저장 (비율만 표시)
        ripeness_info.append(f"Red: {red_ratio:.2f}%, Orange: {orange_ratio:.2f}%, Yellow: {yellow_ratio:.2f}%, Green: {green_ratio:.2f}%")

        # 바운딩 박스를 이미지에 그림 (각 색상에 따라 다르게 표시)
        if red_ratio > max(orange_ratio, yellow_ratio, green_ratio):
            box_color = (0, 0, 255)  # 빨간색
            label = f"R: {red_ratio:.2f}%"
        elif orange_ratio > max(red_ratio, yellow_ratio, green_ratio):
            box_color = (0, 165, 255)  # 주황색
            label = f"O: {orange_ratio:.2f}%"
        elif yellow_ratio > max(red_ratio, orange_ratio, green_ratio):
            box_color = (0, 255, 255)  # 노란색
            label = f"Y: {yellow_ratio:.2f}%"
        else:
            box_color = (0, 255, 0)  # 초록색
            label = f"G: {green_ratio:.2f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # 결과 이미지 저장
    output_path = os.path.join(os.path.dirname(image_path), '결과' + os.path.basename(image_path))
    success = cv2.imwrite(output_path, img)
    return ripeness_info

# 예시 사용법
if __name__ == '__main__':
    image_path = r'C:\Users\ych61\Downloads\tomato.png'  # 처리할 이미지 경로를 지정하세요
    ripeness_results = process_image(image_path)
    for result in ripeness_results:
        print(result)
