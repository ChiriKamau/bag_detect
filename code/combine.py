import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os, csv, time
from datetime import datetime
from pathlib import Path
from collections import Counter

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH   = "/home/test/bag_detect/models/best_float32.tflite"
OUTPUT_DIR   = "/home/test/bag_detect/output"
CAPTURE_DIR  = "/home/test/bag_detect/captures"

CLASSES      = ["backpack", "satchel", "trolley case", "tote bag"]
COLORS       = {
    "backpack":     (255,   0,   0),
    "satchel":      (  0, 255,   0),
    "trolley case": (  0,   0, 255),
    "tote bag":     (  0, 165, 255),
}
CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD        = 0.40
CAPTURE_INTERVAL     = 2  # seconds
# ==========================================

# --- Load model once ---
print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()
model_h, model_w = inp[0]['shape'][1:3]
print(f"Model ready — input: {model_w}x{model_h}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)
csv_out = f"{OUTPUT_DIR}/detections.csv"


def run_inference(image):
    h0, w0 = image.shape[:2]

    # Preprocess
    blob = cv2.resize(image, (model_w, model_h))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = np.expand_dims(blob, axis=0)

    if inp[0]['dtype'] == np.int8:
        scale, zero = inp[0]['quantization']
        blob = (blob.astype(np.float32) / 255.0 / scale + zero).astype(np.int8)
    else:
        blob = blob.astype(np.float32) / 255.0

    # Inference
    t0 = time.time()
    interpreter.set_tensor(inp[0]['index'], blob)
    interpreter.invoke()
    inference_ms = int((time.time() - t0) * 1000)

    # Decode output
    pred = interpreter.get_tensor(out[0]['index'])
    if out[0]['dtype'] == np.int8:
        scale, zero = out[0]['quantization']
        pred = (pred.astype(np.float32) - zero) * scale
    if pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)
    pred = pred[0]

    boxes, scores, class_ids = [], [], []
    normalized = float(pred[0][0]) < 2.0

    for p in pred:
        s = float(np.max(p[4:]))
        if s < CONFIDENCE_THRESHOLD:
            continue
        cid = int(np.argmax(p[4:]))
        cx, cy, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if normalized:
            cx, cy, bw, bh = cx*w0, cy*h0, bw*w0, bh*h0
        else:
            cx = cx*(w0/model_w); cy = cy*(h0/model_h)
            bw = bw*(w0/model_w); bh = bh*(h0/model_h)
        boxes.append([int(cx-bw/2), int(cy-bh/2), int(bw), int(bh)])
        scores.append(s)
        class_ids.append(cid)

    # NMS
    detections = []
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            detections = idxs.flatten().tolist()

    # Draw & collect results
    final_img     = image.copy()
    total         = 0
    class_counter = Counter()
    log           = []

    for i in detections:
        x, y, w, h = boxes[i]
        x = max(0, x); y = max(0, y)
        w = min(w, w0-x); h = min(h, h0-y)
        if w < 15 or h < 15:
            continue

        total += 1
        cls   = CLASSES[class_ids[i]]
        conf  = scores[i]
        class_counter[cls] += 1
        log.append({"id": total, "class": cls, "conf": round(conf*100, 1)})

        color      = COLORS.get(cls, (200, 200, 200))
        thickness  = max(2, w0//400)
        font_scale = max(0.4, w0/1600)
        text       = f"{cls} {int(conf*100)}%"

        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y-10 if y-10 > th else y+th+10
        cv2.rectangle(final_img, (x, ty-th-bl), (x+tw+6, ty+bl), color, -1)
        cv2.rectangle(final_img, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(final_img, text, (x+3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, thickness//2), cv2.LINE_AA)

    cv2.putText(final_img, f"{inference_ms}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return final_img, total, class_counter, log, inference_ms


# --- Main loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not access webcam.")

print(f"Running — capturing every {CAPTURE_INTERVAL}s. Press Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, retrying...")
            time.sleep(1)
            continue

        final_img, total, class_counter, log, inference_ms = run_inference(frame)

        if total > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_out   = f"{OUTPUT_DIR}/{timestamp}_output.jpg"
            cv2.imwrite(img_out, final_img)

            # Append to CSV
            file_exists = os.path.isfile(csv_out)
            with open(csv_out, "a", newline="") as f:
                cols   = ["timestamp", "inference_ms", "total"] + CLASSES
                writer = csv.DictWriter(f, fieldnames=cols)
                if not file_exists:
                    writer.writeheader()
                row = {"timestamp": timestamp, "inference_ms": inference_ms, "total": total}
                for cls in CLASSES:
                    row[cls] = class_counter.get(cls, 0)
                writer.writerow(row)

            print(f"[{timestamp}] {total} bag(s) detected | {inference_ms}ms → saved {img_out}")
            for d in log:
                print(f"  #{d['id']} {d['class']} {d['conf']}%")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No bags detected | {inference_ms}ms")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()