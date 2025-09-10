from flask import Flask, render_template, request, Response, url_for
from ultralytics import YOLO
from PIL import Image
import datetime, os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = 'yolov8n.pt'  # Adjust if needed
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # {id: class_name}

def unique_name(prefix, orig_filename):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base, ext = os.path.splitext(orig_filename)
    return f"{prefix}_{ts}{ext or '.jpg'}"

# ------------------ Video Frame Generator ------------------
def generate_video_frames(video_source, conf, iou):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_source}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            results = model.predict(source=frame_bgr, conf=conf, iou=iou, verbose=False)
            r = results[0]
            annotated = r.plot()  # BGR
            ok, buffer = cv2.imencode('.jpg', annotated)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# ------------------ Routes ------------------
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/exit')
def exit_page():
    return render_template('exit.html')

@app.route('/webcam')
def webcam_page():
    conf = float(request.args.get('conf', 0.25))
    iou = float(request.args.get('iou', 0.45))
    return render_template('webcam.html', conf=conf, iou=iou)

@app.route('/video_feed')
def video_feed():
    conf = float(request.args.get('conf', 0.25))
    iou = float(request.args.get('iou', 0.45))
    cam_index = int(request.args.get('cam', 0))
    return Response(generate_video_frames(cam_index, conf, iou),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_video')
def stream_video():
    video_file = request.args.get('video_file')
    conf = float(request.args.get('conf', 0.25))
    iou = float(request.args.get('iou', 0.45))

    if not video_file:
        return "Missing video file path", 400

    safe_video_file = secure_filename(os.path.basename(video_file))
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_video_file)

    if not os.path.isfile(video_path):
        return f"Video not found: {video_path}", 404

    return Response(generate_video_frames(video_path, conf, iou),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return "No selected files", 400

    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    results_data = []

    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        in_name = unique_name('in', filename)
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], in_name)
        file.save(in_path)
        ext = os.path.splitext(filename)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']

        if not is_video:
            results = model.predict(source=in_path, conf=conf, iou=iou, verbose=False)
            r = results[0]
            plotted_bgr = r.plot()
            plotted_rgb = plotted_bgr[:, :, ::-1]
            out_name = unique_name('out', filename)
            out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
            Image.fromarray(plotted_rgb).save(out_path)

            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                for i in range(len(cls)):
                    dets.append({
                        'class_id': int(cls[i]),
                        'class_name': CLASS_NAMES.get(int(cls[i]), str(int(cls[i]))),
                        'confidence': float(confs[i]),
                        'x1': float(xyxy[i][0]), 'y1': float(xyxy[i][1]),
                        'x2': float(xyxy[i][2]), 'y2': float(xyxy[i][3]),
                    })
            results_data.append({
                'upload_name': filename,
                'image_path': '/' + out_path.replace('\\', '/'),
                'conf': conf,
                'iou': iou,
                'detections': dets,
                'is_video': False
            })
        else:
            results_data.append({
                'upload_name': filename,
                'video_path': '/' + in_path.replace('\\', '/'),
                'conf': conf,
                'iou': iou,
                'detections': [],
                'is_video': True
            })
    return render_template('result.html', results=results_data)

if __name__ == '__main__':
    app.run(debug=True)
