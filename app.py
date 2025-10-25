import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, url_for, redirect, flash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = 'f3956c777cebc566ffb95408917364c2'

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/tomato_effnet.pth'
DISEASE_JSON_PATH = 'tomato_disease_info.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Mail Configuration ===
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'v9630094@gmail.com'
app.config['MAIL_PASSWORD'] = 'rtwt opco zptt jwvz'
mail = Mail(app)

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def normalize_key(name):
    return ''.join(e.lower() for e in name.strip() if e.isalnum())

with open(DISEASE_JSON_PATH, 'r') as f:
    raw_disease_details = json.load(f)
disease_details = {normalize_key(k): v for k, v in raw_disease_details.items()}

REGION_GRID = (2, 2)

# === Daily stats ===
daily_stats = {"count": 0, "timestamps": []}

def log_click():
    daily_stats["count"] += 1
    daily_stats["timestamps"].append(datetime.now().strftime("%H:%M:%S"))

def send_daily_report():
    if daily_stats["count"] == 0:
        return
    try:
        msg = Message(
            "📊 Daily Click Report - Tomato Disease Detection",
            sender=app.config['MAIL_USERNAME'],
            recipients=['tdaitech@gmail.com']
        )
        times = "\n".join(daily_stats["timestamps"])
        msg.body = f"Total Clicks Today: {daily_stats['count']}\n\nTimes:\n{times}"
        mail.send(msg)
        daily_stats["count"] = 0
        daily_stats["timestamps"] = []
    except Exception as e:
        print("❌ Error sending daily report:", e)

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(send_daily_report, 'cron', hour=23, minute=59)
scheduler.start()

def split_image_regions(image, grid=(2, 2)):
    w, h = image.size
    ws = w // grid[0]; hs = h // grid[1]
    regions = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            left = i * ws; top = j * hs
            regions.append(image.crop((left, top, left + ws, top + hs)))
    return regions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    log_click()
    if 'image' not in request.files or request.files['image'].filename == '':
        flash("❌ No image uploaded", "danger")
        return redirect('/')
    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = Image.open(path).convert('RGB')
    subs = split_image_regions(img, REGION_GRID)

    best_conf = 0
    best_label = None
    for r in subs:
        t = transform(r).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.nn.functional.softmax(model(t), dim=1)
            conf, pred = torch.max(p, 1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_label = class_names[pred.item()].strip()

    if not best_label:
        flash("⚠️ Could not classify image", "warning")
        return redirect('/')

    label = "Healthy" if best_label.lower() == "healthy" else best_label
    d = disease_details.get(normalize_key(label), {})

    info = [{"label": label, "details": {
        "explanation": d.get("explanation", f"Detected {label}."),
        "water": d.get("water", "N/A"),
        "fertilizer": d.get("fertilizer", "N/A"),
        "medicine": d.get("medicine", ["N/A"]),
        "organic_medicine": d.get("organic_medicine", ["N/A"]),
        "prevention": d.get("prevention", "N/A")
    }}]

    return render_template('index.html',
                           multi_predictions=info,
                           image_url=url_for('static', filename='uploads/' + filename))

@app.route('/predict_video', methods=['POST'])
def predict_video():
    log_click()
    if 'video' not in request.files or request.files['video'].filename == '':
        flash("❌ No video uploaded", "danger")
        return redirect('/')
    f = request.files['video']
    name = secure_filename(f.filename)
    vp = os.path.join(UPLOAD_FOLDER, name)
    f.save(vp)

    cap = cv2.VideoCapture(vp)
    fr = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fr) if fr > 0 else 10

    preds = []
    i = 0
    while cap.isOpened():
        r, frm = cap.read()
        if not r:
            break
        if i % interval == 0:
            g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            if 40 < np.mean(g) < 220:
                pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                for s in split_image_regions(pil, REGION_GRID):
                    t = transform(s).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = torch.nn.functional.softmax(model(t), dim=1)
                        _, pr = torch.max(p, 1)
                        preds.append(class_names[pr.item()].strip())
        i += 1
    cap.release()
    os.remove(vp)

    if not preds:
        flash("⚠️ No disease found", "warning")
        return redirect('/')

    c = Counter(preds)
    mc = [l for l, n in c.items() if n >= 2] or list(c.keys())
    if any(l.lower() == "healthy" for l in mc):
        mc = ["Healthy"]

    info = []
    for l in mc:
        d = disease_details.get(normalize_key(l), {})
        info.append({"label": l, "details": {
            "explanation": d.get("explanation", f"Detected {l}."),
            "water": d.get("water", "N/A"),
            "fertilizer": d.get("fertilizer", "N/A"),
            "medicine": d.get("medicine", ["N/A"]),
            "organic_medicine": d.get("organic_medicine", ["N/A"]),
            "prevention": d.get("prevention", "N/A")
        }})

    return render_template('index.html', multi_predictions=info, image_url=None)

@app.route('/send_email', methods=['POST'])
def send_email():
    log_click()
    name = request.form.get('name')
    email = request.form.get('email')
    msgt = request.form.get('message')
    photo = request.files.get('photo')
    if not (name and email and msgt):
        flash("❗ Fill all fields", "warning")
        return redirect('/')
    try:
        m = Message("🌿 New Contact Request",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=['tdaitech@gmail.com'])
        m.body = f"Name:{name}\nEmail:{email}\nMessage:{msgt}"
        if photo and photo.filename:
            fn = secure_filename(photo.filename)
            fp = os.path.join(UPLOAD_FOLDER, fn)
            photo.save(fp)
            with open(fp, 'rb') as f:
                m.attach(fn, "image/jpeg", f.read())
        mail.send(m)
        r = Message("✅ Thank you!", sender=app.config['MAIL_USERNAME'], recipients=[email])
        r.body = f"Hi {name},\nWe received your message."
        mail.send(r)
        flash("✅ Message sent!", "success")
    except Exception as e:
        print(e)
        flash("❌ Failed to send", "danger")
    return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
