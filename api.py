from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
import math

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

@app.get("/")
def health():
    return {"status": "Face Ratio API running"}

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):

    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    h, w = image.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark

    mouth_width = distance(landmarks[61], landmarks[291])
    nose_width = distance(landmarks[94], landmarks[331])
    left_eye_width = distance(landmarks[33], landmarks[133])
    right_eye_width = distance(landmarks[362], landmarks[263])
    eye_spacing = distance(landmarks[133], landmarks[362])
    face_width = distance(landmarks[234], landmarks[454])
    face_height = distance(landmarks[10], landmarks[152])

    if face_width == 0 or nose_width == 0:
        return {"error": "Face landmarks not reliable"}

    face_ratio = face_height / face_width
    eye_spacing_ratio = eye_spacing / ((left_eye_width + right_eye_width) / 2)
    mouth_nose_ratio = mouth_width / nose_width

    return {
        "face_ratio": round(face_ratio, 3),
        "eye_spacing_ratio": round(eye_spacing_ratio, 3),
        "mouth_nose_ratio": round(mouth_nose_ratio, 3)
    }
