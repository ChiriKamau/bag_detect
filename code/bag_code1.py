import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os, csv, time
from pathlib import Path
from collections import Counter

# ==========================================
# CONFIG — edit these
# ==========================================
MODEL_PATH = "/home/test/bag_detect/models/best_dynamic_range_quant.tflite"
IMAGE_PATH = "/home/test/bag_detect/images/image_20260220_120854.jpg"
OUTPUT_DIR = "/home/test/bag_detct/output"

CLASSES = ["backpack", "satchel", "trolley case", "tote bag"]
COLORS  = {
    "backpack":     (255,   0,   0),
    "satchel":      (  0, 255,   0),
    "trolley case": (  0,   0, 255),
    "tote bag":     (  0, 165, 255),
}
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD        = 0.40
# ==========================================

# Load model
print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()
model_h, model_w = inp[0]['shape'][1:3]
print(f"Model ready — input: {model_w}x{model_h}")

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError(f"Image not found: {IMAGE_PATH}")
h0, w0    = image.shape[:2]
final_img = image.copy()
print(f"Image loaded: {w0}x{h0}")

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
print("Running inference...")
t0 = time.time()
interpreter.set_tensor(inp[0]['index'], blob)
interpreter.invoke()
inference_ms = int((time.time() - t0) * 1000)
print(f"Inference time: {inference_ms} ms")

# Get output
pred = interpreter.get_tensor(out[0]['index'])
if out[0]['dtype'] == np.int8:
    scale, zero = out[0]['quantization']
    pred = (pred.astype(np.float32) - zero) * scale

if pred.shape[1] < pred.shape[2]:
    pred = pred.transpose(0, 2, 1)
pred = pred[0]

# Decode boxes
boxes, scores, class_ids = [], [], []
normalized = float(pred[0][0]) < 2.0

for p in pred:
    s = float(np.max(p[4:]))
    if s < CONFIDENCE_THRESHOLD:
        continue
    cid          = int(np.argmax(p[4:]))
    cx, cy, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if normalized:
        cx, cy, w, h = cx*w0, cy*h0, w*w0, h*h0
    else:
        cx = cx*(w0/model_w); cy = cy*(h0/model_h)
        w  = w *(w0/model_w); h  = h *(h0/model_h)
    boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
    scores.append(s)
    class_ids.append(cid)

# NMS
detections = []
if boxes:
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(idxs) > 0:
        detections = idxs.flatten().tolist()

# Draw & count
total         = 0
class_counter = Counter()
log           = []

for i in detections:
    x, y, w, h = boxes[i]
    x = max(0,x); y = max(0,y)
    w = min(w, w0-x); h = min(h, h0-y)
    if w < 15 or h < 15:
        continue

    total += 1
    cls   = CLASSES[class_ids[i]]
    conf  = scores[i]
    class_counter[cls] += 1
    log.append({"id": total, "class": cls, "conf": round(conf*100, 1)})

    color      = COLORS.get(cls, (200,200,200))
    thickness  = max(2, w0//400)
    font_scale = max(0.4, w0/1600)
    text       = f"{cls} {int(conf*100)}%"

    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    ty = y-10 if y-10 > th else y+th+10
    cv2.rectangle(final_img, (x, ty-th-bl), (x+tw+6, ty+bl), color, -1)
    cv2.rectangle(final_img, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(final_img, text, (x+3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255,255,255), max(1, thickness//2), cv2.LINE_AA)

cv2.putText(final_img, f"{inference_ms}ms", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
base    = Path(IMAGE_PATH).stem
img_out = f"{OUTPUT_DIR}/{base}_output.jpg"
csv_out = f"{OUTPUT_DIR}/detections.csv"
cv2.imwrite(img_out, final_img)

exists = os.path.isfile(csv_out)
with open(csv_out, "a", newline="") as f:
    cols   = ["image", "inference_ms", "total"] + CLASSES
    writer = csv.DictWriter(f, fieldnames=cols)
    if not exists:
        writer.writeheader()
    row = {"image": Path(IMAGE_PATH).name, "inference_ms": inference_ms, "total": total}
    for cls in CLASSES:
        row[cls] = class_counter.get(cls, 0)
    writer.writerow(row)

# Summary
print("\n========== DETECTION SUMMARY ==========")
print(f"  Total bags     : {total}")
print(f"  Inference time : {inference_ms} ms")
for cls in CLASSES:
    c = class_counter.get(cls, 0)
    print(f"  {cls:<15} : {c}  {'█'*c}")
print("\n--- Per Detection ---")
for d in log:
    print(f"  #{d['id']} | {d['class']:<15} | {d['conf']}%")
print("========================================")
print(f"\n  Image saved → {img_out}")
print(f"  CSV updated → {csv_out}")