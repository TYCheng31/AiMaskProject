from flask import Flask, render_template, Response, jsonify, request
import requests  # <-- 記得安裝 requests: pip install requests
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# 請將你的 Line Notify Token 放在這裡
LINE_NOTIFY_TOKEN = "L4r3y6YeL5aV2q6svJJdvk1YAcm1YxViSAQC61lNPeG"
LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"

# YOLO 模型
model = YOLO(r"E:\AIMaskProject\best.pt")
cap = cv2.VideoCapture(0)

g_counts = {}
g_no_mask_coords = []

def gen_frames():
    global g_counts, g_no_mask_coords

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.5, imgsz=640)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        counts = {}
        no_mask_coords = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name not in counts:
                counts[class_name] = 1
            else:
                counts[class_name] += 1

            if class_name == "No_Mask":
                no_mask_coords.append((int(x1), int(y1)))

        g_counts = counts
        g_no_mask_coords = no_mask_coords

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # 你的前端頁面

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    data = {
        "counts": g_counts,
        "no_mask_coords": g_no_mask_coords
    }
    return jsonify(data)


@app.route('/line_notify', methods=['POST'])
def line_notify():
    """
    接收前端送來的 JSON，
    其中包含 { "message": "...想發送的訊息..." }
    並呼叫 Line Notify API 發送到你的群組。
    """
    req_data = request.json or {}
    message = req_data.get("message", "No message")

    headers = {
        "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"
    }
    payload = {
        "message": message
    }

    # 發送 POST 請求給 Line Notify
    r = requests.post(LINE_NOTIFY_API, headers=headers, data=payload)
    if r.status_code == 200:
        return "Line message sent!", 200
    else:
        return f"Failed to send message: {r.text}", 400


if __name__ == '__main__':
    # 若想讓手機同網路下可連線，使用 host='0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)
