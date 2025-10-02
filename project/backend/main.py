from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import base64
import json

app = FastAPI(title="AI Physiotherapy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EXERCISE_PLANS = {
    "shoulder injury": {
        "ailment": "shoulder injury",
        "exercises": [
            {
                "name": "Shoulder Flexion",
                "description": "Raise your arm forward and up",
                "target_reps": 12,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Shoulder Abduction",
                "description": "Raise your arm out to the side",
                "target_reps": 12,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Shoulder Pendulum",
                "description": "Gently swing your arm in small circles",
                "target_reps": 10,
                "sets": 3,
                "rest_seconds": 30
            }
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 6
    },
    "elbow injury": {
        "ailment": "elbow injury",
        "exercises": [
            {
                "name": "Elbow Flexion",
                "description": "Bend your elbow bringing hand toward shoulder",
                "target_reps": 15,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Elbow Extension",
                "description": "Straighten your elbow completely",
                "target_reps": 15,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Wrist Rotation",
                "description": "Rotate your wrist palm up and down",
                "target_reps": 12,
                "sets": 3,
                "rest_seconds": 30
            }
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 4
    },
    "wrist injury": {
        "ailment": "wrist injury",
        "exercises": [
            {
                "name": "Wrist Flexion",
                "description": "Bend your wrist forward and back",
                "target_reps": 15,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Wrist Extension",
                "description": "Extend your wrist upward",
                "target_reps": 15,
                "sets": 3,
                "rest_seconds": 30
            },
            {
                "name": "Wrist Circles",
                "description": "Make circular motions with your wrist",
                "target_reps": 10,
                "sets": 3,
                "rest_seconds": 30
            }
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 3
    }
}

class AilmentRequest(BaseModel):
    ailment: str

class FrameRequest(BaseModel):
    frame: str
    exercise_name: str
    previous_state: Optional[dict] = None

class SessionResult(BaseModel):
    reps: int
    feedback: List[dict]
    accuracy_score: float
    state: dict

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_shoulder_flexion(landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    angle = calculate_angle(hip, shoulder, elbow)

    feedback = []
    if angle < 80:
        feedback.append({"type": "correction", "message": "Raise your arm higher"})
    elif angle > 160:
        feedback.append({"type": "encouragement", "message": "Great form! Full range achieved"})

    return angle, feedback

def analyze_elbow_flexion(landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    angle = calculate_angle(shoulder, elbow, wrist)

    feedback = []
    if angle > 150:
        feedback.append({"type": "correction", "message": "Bend your elbow more"})
    elif angle < 50:
        feedback.append({"type": "encouragement", "message": "Perfect! Full flexion achieved"})

    return angle, feedback

@app.get("/")
def root():
    return {"message": "AI Physiotherapy API is running", "status": "healthy"}

@app.post("/api/get_plan")
def get_exercise_plan(request: AilmentRequest):
    ailment = request.ailment.lower()

    if ailment in EXERCISE_PLANS:
        return EXERCISE_PLANS[ailment]

    available = list(EXERCISE_PLANS.keys())
    raise HTTPException(
        status_code=404,
        detail=f"Exercise plan not found for '{ailment}'. Available plans: {available}"
    )

@app.post("/api/analyze_frame")
def analyze_frame(request: FrameRequest):
    try:
        img_data = base64.b64decode(request.frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return {
                "reps": request.previous_state.get("reps", 0) if request.previous_state else 0,
                "feedback": [{"type": "warning", "message": "No pose detected. Please stand in view of camera"}],
                "accuracy_score": 0.0,
                "state": request.previous_state or {"reps": 0, "stage": "down"}
            }

        landmarks = results.pose_landmarks.landmark
        exercise_name = request.exercise_name.lower()

        current_state = request.previous_state or {"reps": 0, "stage": "down"}
        reps = current_state.get("reps", 0)
        stage = current_state.get("stage", "down")

        if "shoulder" in exercise_name:
            angle, feedback = analyze_shoulder_flexion(landmarks)

            if angle > 140 and stage == "down":
                stage = "up"
            if angle < 80 and stage == "up":
                stage = "down"
                reps += 1
                feedback.append({"type": "encouragement", "message": f"Rep {reps} completed!"})

        elif "elbow" in exercise_name:
            angle, feedback = analyze_elbow_flexion(landmarks)

            if angle < 60 and stage == "down":
                stage = "up"
            if angle > 140 and stage == "up":
                stage = "down"
                reps += 1
                feedback.append({"type": "encouragement", "message": f"Rep {reps} completed!"})
        else:
            angle = 0
            feedback = [{"type": "warning", "message": "Exercise not recognized"}]

        accuracy = min(100.0, (reps / 10.0) * 100) if reps > 0 else 0.0

        return {
            "reps": reps,
            "feedback": feedback,
            "accuracy_score": round(accuracy, 2),
            "state": {"reps": reps, "stage": stage, "angle": angle}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/api/progress/{user_id}")
def get_progress(user_id: str):
    return {
        "user_id": user_id,
        "total_sessions": 12,
        "total_reps": 450,
        "average_accuracy": 87.5,
        "streak_days": 5,
        "weekly_data": [
            {"day": "Mon", "reps": 60, "accuracy": 85},
            {"day": "Tue", "reps": 70, "accuracy": 88},
            {"day": "Wed", "reps": 65, "accuracy": 86},
            {"day": "Thu", "reps": 75, "accuracy": 90},
            {"day": "Fri", "reps": 80, "accuracy": 89},
            {"day": "Sat", "reps": 55, "accuracy": 84},
            {"day": "Sun", "reps": 45, "accuracy": 82}
        ],
        "recent_sessions": [
            {
                "date": "2025-10-01",
                "exercise": "Shoulder Flexion",
                "reps": 12,
                "accuracy": 89
            },
            {
                "date": "2025-09-30",
                "exercise": "Elbow Flexion",
                "reps": 15,
                "accuracy": 92
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
