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

    # 각 바운딩 박스 내에서 색상 비율 계산 및 외곽선 추적
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])

        # 바운딩 박스 내 색상 비율 계산
        box_region = hsv_img[y1:y2, x1:x2]

        # HSV 마스크를 이용하여 토마토 영역만 추출 (폴리곤 영역만 활용)
        mask = None
        box_img = hsv_img[y1:y2, x1:x2]

        # 이미지 전처리 (블러링 및 이진화)
        blurred = cv2.GaussianBlur(box_img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이진화 처리

        # 외곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 외곽선을 그려 폴리곤 생성
        for contour in contours:
            # 작은 외곽선 필터링 (너무 작은 것은 무시)
            if cv2.contourArea(contour) < 500:  # 작은 잡음을 걸러냄 (기존 100에서 500으로 증가)
                continue

            # 외곽선 단순화
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 폴리곤 마스크 생성
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)

            # 폴리곤 마스크를 적용하여 해당 영역만 추출
            polygon_region = cv2.bitwise_and(box_img, box_img, mask=mask)

            # 폴리곤 영역에서 색상 비율 계산
            red_mask_poly = cv2.inRange(polygon_region, red_lower1, red_upper1) + cv2.inRange(polygon_region, red_lower2, red_upper2)
            polygon_total_pixels = np.sum(mask > 0)
            if polygon_total_pixels > 0:
                red_ratio_poly = (np.sum(red_mask_poly > 0) / polygon_total_pixels) * 100
                label_poly = f"Poly R: {red_ratio_poly:.2f}%"
                ripeness_info.append(label_poly)
                # 폴리곤 비율도 이미지에 한 번만 표시
                cv2.putText(img, label_poly, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break  # 한 폴리곤에 대해서만 처리하고 중복 계산 방지

            # 실제 윤곽선 그리기 (폴리곤 형태로)
            cv2.drawContours(img[y1:y2, x1:x2], [approx], -1, (255, 0, 0), 2)  # 파란색으로 외곽선을 그림

    # 결과 이미지 저장
    output_path = os.path.join(os.path.dirname(image_path), '결과_' + os.path.basename(image_path))
    success = cv2.imwrite(output_path, img)
    return ripeness_info

# 예시 사용법
if __name__ == '__main__':
    image_path = r'C:\Users\ych61\Downloads\tomato.png'  # 처리할 이미지 경로를 지정하세요
    ripeness_results = process_image(image_path)
    for result in ripeness_results:
        print(result)
