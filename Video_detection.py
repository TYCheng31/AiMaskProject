import cv2
from ultralytics import YOLO

# 載入訓練好的 YOLOv8 模型
model = YOLO('D:\\AiMaskProject\\best.pt')  # 替換為您的模型路徑

# 指定影片檔案的路徑
video_path = 'C:\\Users\\littl\\Videos\\2025-01-06 12-29-55.mkv'  # 替換為您的影片檔案路徑

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功開啟
if not cap.isOpened():
    print(f"無法開啟影片檔案: {video_path}")
    exit()

# 獲取影片的 FPS
fps = cap.get(cv2.CAP_PROP_FPS)
fps_text = f"FPS: {fps:.2f}"

while True:
    ret, frame = cap.read()
    if not ret:
        print("影片播放結束或無法讀取影片幀")
        break

    # 使用 YOLOv8 模型進行預測
    results = model(frame, conf=0.5)  # conf 設定信心門檻

    # 繪製預測結果
    annotated_frame = results[0].plot()

    # 在左上角加入影片 FPS
    cv2.putText(
        annotated_frame,                    # 圖像
        fps_text,                           # 顯示文字
        (10, 30),                           # 文字位置 (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,           # 字體
        1,                                  # 字體大小
        (0, 255, 0),                        # 顏色 (綠色)
        2,                                  # 字體粗細
        cv2.LINE_AA                         # 線條類型
    )

    # 顯示影像
    cv2.imshow('Mask Detection', annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
