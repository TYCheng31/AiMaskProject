from flask import Flask, render_template, Response, jsonify, request
import requests
import cv2
from ultralytics import YOLO

# ==== MongoDB 相關套件 ====
from pymongo import MongoClient
from datetime import datetime
import time

# === 新增，用於接收並合併 Base64 圖片 ===
import base64
from PIL import Image
import io

app = Flask(__name__)

# ==== (A) MongoDB 連線設定 ====
client = MongoClient("mongodb://localhost:27017/")
db = client["mask_database"]           # 資料庫名稱
collection = db["mask_detections"]     # 集合名稱

# ==== (B) Line Notify 設定 ====
LINE_NOTIFY_TOKEN = "L4r3y6YeL5aV2q6svJJdvk1YAcm1YxViSAQC61lNPeG"
LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"

# ==== (C) YOLO 模型 & 攝影機設定 ====
model = YOLO("D:\\AiMaskProject\\bestm.pt")  # 請改為實際模型路徑

# 將原先攝影機輸入改為影片檔案輸入
cap = cv2.VideoCapture("C:\\Users\\littl\\Videos\\2025-01-06 13-12-39.mkv")

# ==== (D) 全域變數：用於前端顯示 ====
g_counts = {}            # 用於儲存各類別人數 (e.g. {"Mask": x, "No_Mask": y})
g_no_mask_coords = []    # 用於儲存未戴口罩座標
g_fps = 0                # 用於儲存即時 FPS

def gen_frames():
    """
    每當前端請求 /video_feed 時，就會持續執行此函式，
    逐幀讀取影像並透過 yield 傳回串流。
    """
    global g_counts, g_no_mask_coords, g_fps

    # 紀錄上次寫入資料庫的時間（秒數）
    last_saved_time = 0

    # 初始化 FPS 計算
    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 使用 YOLO 進行偵測
        results = model.predict(source=frame, conf=0.5, imgsz=640)
        annotated_frame = results[0].plot()

        # 解析偵測結果
        boxes = results[0].boxes
        counts = {}
        no_mask_coords = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            # 計算該 class_name 累計人數
            counts[class_name] = counts.get(class_name, 0) + 1

            # 若是未戴口罩，記錄座標
            if class_name == "No_Mask":
                no_mask_coords.append((int(x1), int(y1)))

        # 更新全域變數，以供前端查詢
        g_counts = counts
        g_no_mask_coords = no_mask_coords

        # ============ (1) 計算要儲存的數據 ============
        total_people = sum(counts.values())      # 總人數
        mask_count = counts.get("Mask", 0)       # 戴口罩人數
        no_mask_count = counts.get("No_Mask", 0) # 未戴口罩人數
        if total_people > 0:
            no_mask_percent = round((no_mask_count / total_people) * 100, 2)
            mask_percent = round(100 - no_mask_percent, 2)
        else:
            no_mask_percent = 0.0
            mask_percent = 0.0

        # ============ (2) 每 5 秒才存一次 並且有偵測到人 ============
        now = time.time()
        if now - last_saved_time >= 5 and total_people > 0:
            record = {
                "timestamp": datetime.now(),
                "total_people": total_people,
                "mask_count": mask_count,
                "no_mask_count": no_mask_count,
                "no_mask_percent": no_mask_percent,
                "mask_percent": mask_percent
            }
            collection.insert_one(record)
            print("已寫入資料庫:", record)

            # 更新最後寫入時間
            last_saved_time = now

        # ============ (3) 計算 FPS ============
        frame_count += 1
        elapsed_time = now - start_time
        if elapsed_time >= 1.0:
            g_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = now

        # 影像轉為串流回應給前端
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')

# ==== (E) Flask 路由 ====
@app.route('/')
def index():
    """
    前端首頁 (index.html)，顯示即時影像與其他資訊
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    影像串流端點，前端 <img src="/video_feed"> 就能顯示即時畫面
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    """
    前端每秒呼叫此端點，取得目前偵測結果 (json 格式)
    """
    data = {
        "counts": g_counts,
        "no_mask_coords": g_no_mask_coords,
        "fps": g_fps  # 新增 FPS
    }
    return jsonify(data)

@app.route('/line_notify', methods=['POST'])
def line_notify():
    """
    從前端接收 JSON ({"message": "..."}),
    呼叫 LINE Notify API 將訊息發送到指定群組/聊天室
    """
    req_data = request.json or {}
    message = req_data.get("message", "No message")

    headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}
    payload = {"message": message}
    r = requests.post(LINE_NOTIFY_API, headers=headers, data=payload)

    if r.status_code == 200:
        return "Line message sent!", 200
    else:
        return f"Failed to send message: {r.text}", 400

@app.route('/save_to_mongo', methods=['POST'])
def save_to_mongo():
    """
    (選擇性) 前端按下「手動儲存」時，立即將目前的偵測結果存到資料庫
    """
    global g_counts, g_no_mask_coords

    total_people = sum(g_counts.values())
    mask_count = g_counts.get("Mask", 0)
    no_mask_count = g_counts.get("No_Mask", 0)
    if total_people > 0:
        no_mask_percent = round((no_mask_count / total_people) * 100, 2)
        mask_percent = round(100 - no_mask_percent, 2)
    else:
        no_mask_percent = 0.0
        mask_percent = 0.0

    # 只有當有偵測到人時才儲存
    if total_people > 0:
        record = {
            "timestamp": datetime.now(),
            "total_people": total_people,
            "mask_count": mask_count,
            "no_mask_count": no_mask_count,
            "no_mask_percent": no_mask_percent,
            "mask_percent": mask_percent
        }
        collection.insert_one(record)
        print("手動儲存成功:", record)

        return jsonify({"status": "success", "message": "手動儲存成功，已寫入 MongoDB"})
    else:
        return jsonify({"status": "no_data", "message": "目前無偵測到人物，未儲存資料。"}), 400

# ==== (F) 查詢資料庫最新 20 筆紀錄 ====
@app.route('/latest_records', methods=['GET'])
def latest_records():
    """
    查詢資料庫最新 20 筆紀錄 (依 timestamp 由新到舊排序)，
    以 JSON 形式返回給前端。
    """
    docs = collection.find().sort("timestamp", -1).limit(20)

    result = []
    for doc in docs:
        record = {
            "timestamp": doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "total_people": doc.get("total_people", 0),
            "mask_count": doc.get("mask_count", 0),
            "no_mask_count": doc.get("no_mask_count", 0),
            "no_mask_percent": doc.get("no_mask_percent", 0.0),
            "mask_percent": doc.get("mask_percent", 0.0)
        }
        result.append(record)

    return jsonify(result)

# ==== (G) ★ 新增：接收合併後的圖表 Base64、傳給 LINE Notify ====
@app.route('/send_charts_to_line', methods=['POST'])
def send_charts_to_line():
    """
    接收合併後的圖表的 Base64 字串，並上傳到 Line Notify
    """
    data = request.get_json() or {}
    chart_base64 = data.get('chart', '')

    if not chart_base64:
        return "未接收到圖表資料。", 400

    try:
        # 1) 去除 data:image/png;base64, 的前綴
        chart_base64 = chart_base64.split(',')[1]

        # 2) Base64 解碼成二進位資料
        chart_data = base64.b64decode(chart_base64)

        # 3) 使用 PIL 開啟圖片 (轉為 RGBA 確保格式一致)
        img = Image.open(io.BytesIO(chart_data)).convert('RGBA')

        # 4) 轉成 BytesIO 以供上傳
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # 5) 呼叫 Line Notify API，上傳圖片 (multipart/form-data)
        headers = {
            "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}",
        }
        files = {
            "imageFile": ("chart.png", img_bytes, "image/png")
        }
        payload = {
            "message": "分析報告"
        }
        r = requests.post(LINE_NOTIFY_API, headers=headers, files=files, data=payload)

        if r.status_code == 200:
            return "成功傳送圖表到 LINE Notify!", 200
        else:
            return f"傳送失敗: {r.text}", 400

    except Exception as e:
        print("send_charts_to_line error:", e)
        return "後端處理圖表時發生錯誤。", 500

# ==== (H) 主程式入口 ====
if __name__ == '__main__':
    # 若需讓同網路下的手機也可連線，使用 host='0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)
