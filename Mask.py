import cv2
import numpy as np
from ultralytics import YOLO

# 1. 載入 YOLOv8 模型
#    - **關鍵字**：請確認 'best.pt' 路徑是否正確
model = YOLO("E:\\AIFinalProject\\best.pt")

# 2. 開啟預設鏡頭
#    - **重點**：cap = cv2.VideoCapture(0) 代表使用筆電預設鏡頭
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 模型推論
    #    - **關鍵字**：可調整 conf=0.5（信心閾值）、imgsz=640（輸入影像大小）
    results = model.predict(source=frame, conf=0.5, imgsz=640)

    # 4. 繪製 YOLOv8 偵測框
    annotated_frame = results[0].plot()

    # 5. 統計各類別人數 & 收集「沒戴口罩」的座標
    boxes = results[0].boxes  
    counts = {}               
    no_mask_coords = []       

    for box in boxes:
        # 取得邊界框座標 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0]

        # 取得類別 ID 與類別名稱
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        # 統計類別人數
        if class_name not in counts:
            counts[class_name] = 1
        else:
            counts[class_name] += 1

        # 如果類別為 "No_Mask" 就把座標 (x1, y1) 加到清單
        if class_name == "No_Mask":
            no_mask_coords.append((x1, y1))

    # ----------- 在「第二視窗」顯示資訊 -----------
    # 6. 建立一個空白畫布 (例如 400x400，背景為黑色)
    info_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    # 若想要白色背景，可以改成：info_frame = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # 7. 在 info_frame 上繪製文字
    y_start = 20  # 文字繪製的起始 Y 座標
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_class_count = (0, 255, 0)  # 綠色
    color_no_mask = (0, 0, 255)     # 紅色

    # (A) 「人數分布」標題（始終顯示）
    cv2.putText(
        info_frame,
        "Class",
        (10, y_start),
        font,
        0.7,
        color_class_count,
        2
    )
    y_start += 30

    # 若有偵測到物件，逐條顯示各類別人數；若沒有偵測到，這裡不會印任何類別
    for class_name, count in counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(
            info_frame,
            text,
            (10, y_start),
            font,
            font_scale,
            color_class_count,
            1
        )
        y_start += 20

    # (B) 「未戴口罩位置」標題（始終顯示）
    y_start += 10  # 空出行距
    cv2.putText(
        info_frame,
        "No_Mask Locate:",
        (10, y_start),
        font,
        0.7,
        color_no_mask,
        2
    )
    y_start += 30

    # 若有偵測到 No_Mask，逐條顯示其座標；若沒有，這裡不會印任何座標
    for i, (nx, ny) in enumerate(no_mask_coords, start=1):
        coord_text = f"{i}: ({int(nx)}, {int(ny)})"
        cv2.putText(
            info_frame,
            coord_text,
            (10, y_start),
            font,
            font_scale,
            color_no_mask,
            1
        )
        y_start += 20

    # 8. 顯示兩個視窗：原畫面(含偵測框) & 資訊視窗(統計/座標)
    cv2.imshow("YOLOv8 Webcam - Mask Detection", annotated_frame)
    cv2.imshow("Data Window", info_frame)

    # 9. 按 'ESC' 鍵退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 10. 釋放資源
cap.release()
cv2.destroyAllWindows()
