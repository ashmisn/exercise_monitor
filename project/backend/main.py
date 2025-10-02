from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import cv2
import mediapipe as mp
import numpy as np
import base64
import json

# =========================================================================
# 1. MEDIAPIPE INITIALIZATION (GLOBAL)
# =========================================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================================
# 2. FASTAPI APP & MIDDLEWARE
# =========================================================================
app = FastAPI(title="AI Physiotherapy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 3. DATA MODELS (Pydantic) & Exercise Plans
# =========================================================================
class AilmentRequest(BaseModel):
    ailment: str

class FrameRequest(BaseModel):
    frame: str
    exercise_name: str
    previous_state: Optional[dict] = None

class Landmark2D(BaseModel):
    x: float
    y: float
    visibility: float

class SessionResult(BaseModel):
    reps: int
    feedback: List[Dict]
    accuracy_score: float
    state: Dict
    drawing_landmarks: Optional[List[Landmark2D]] = None # NEW
    current_angle: Optional[float] = None              # NEW
    angle_coords: Optional[Dict] = None                # NEW (A, B, C points for drawing angle arc)

EXERCISE_PLANS = {
    # ... (Your EXERCISE_PLANS dictionary remains here) ...
    "shoulder injury": {
        "ailment": "shoulder injury",
        "exercises": [
            { "name": "Shoulder Flexion", "description": "Raise your arm forward and up", "target_reps": 12, "sets": 3, "rest_seconds": 30 },
            { "name": "Shoulder Abduction", "description": "Raise your arm out to the side", "target_reps": 12, "sets": 3, "rest_seconds": 30 }
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 6
    },
    "elbow injury": {
        "ailment": "elbow injury",
        "exercises": [
            { "name": "Elbow Flexion", "description": "Bend your elbow bringing hand toward shoulder", "target_reps": 15, "sets": 3, "rest_seconds": 30 },
            { "name": "Elbow Extension", "description": "Straighten your elbow completely", "target_reps": 15, "sets": 3, "rest_seconds": 30 }
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 4
    }
}


# =========================================================================
# 4. UTILITY FUNCTIONS (Simplified for API use)
# =========================================================================

def calculate_angle_2d(a, b, c):
    """Calculates angle using 2D coordinates (better for visual feedback on screen)"""
    a = np.array(a) # Point A (e.g., Hip/Shoulder)
    b = np.array(b) # Point B (Vertex, e.g., Shoulder/Elbow)
    c = np.array(c) # Point C (e.g., Elbow/Wrist)

    # Use 2D calculation for simpler planar movements
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_2d_landmarks(landmarks):
    """Extracts 2D landmarks for drawing on the frontend canvas."""
    return [
        {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
        for lm in landmarks
    ]

# =========================================================================
# 5. CORE ANALYSIS LOGIC
# =========================================================================

def analyze_shoulder_flexion(landmarks):
    # Landmarks for Shoulder Flexion (Angle at the Shoulder)
    # The points for the angle calculation (Hip-Shoulder-Elbow)
    P_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    P_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    P_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

    angle = calculate_angle_2d(P_HIP, P_SHOULDER, P_ELBOW)
    
    # Coordinates needed for the angle arc drawing on frontend
    angle_coords = {
        "A": {"x": P_HIP[0], "y": P_HIP[1]},
        "B": {"x": P_SHOULDER[0], "y": P_SHOULDER[1]},
        "C": {"x": P_ELBOW[0], "y": P_ELBOW[1]},
    }

    feedback = []
    if angle < 90:
        feedback.append({"type": "correction", "message": "Raise your arm higher (Hip-Shoulder-Elbow angle should be smaller)"})
    elif angle > 160:
        feedback.append({"type": "encouragement", "message": "Arm fully relaxed - ready to lift"})

    return angle, feedback, angle_coords

def analyze_elbow_flexion(landmarks):
    # Landmarks for Elbow Flexion (Angle at the Elbow)
    # The points for the angle calculation (Shoulder-Elbow-Wrist)
    P_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    P_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    P_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    angle = calculate_angle_2d(P_SHOULDER, P_ELBOW, P_WRIST)
    
    # Coordinates needed for the angle arc drawing on frontend
    angle_coords = {
        "A": {"x": P_SHOULDER[0], "y": P_SHOULDER[1]},
        "B": {"x": P_ELBOW[0], "y": P_ELBOW[1]},
        "C": {"x": P_WRIST[0], "y": P_WRIST[1]},
    }

    feedback = []
    if angle > 150:
        feedback.append({"type": "correction", "message": "Bend your elbow more for full range"})
    elif angle < 60:
        feedback.append({"type": "encouragement", "message": "Deep bend achieved! Now extend slowly."})

    return angle, feedback, angle_coords

# =========================================================================
# 6. API ENDPOINTS
# =========================================================================

@app.get("/")
def root():
    return {"message": "AI Physiotherapy API is running", "status": "healthy"}

@app.post("/api/get_plan")
def get_exercise_plan(request: AilmentRequest):
    # ... (function body remains the same) ...
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
        # 1. Decode Frame
        img_data = base64.b64decode(request.frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        current_state = request.previous_state or {"reps": 0, "stage": "down"}
        reps = current_state.get("reps", 0)
        stage = current_state.get("stage", "down")
        
        # --- Handle No Pose Detected ---
        if not results.pose_landmarks:
            return {
                "reps": reps,
                "feedback": [{"type": "warning", "message": "No pose detected. Adjust camera view."}],
                "accuracy_score": 0.0,
                "state": current_state
            }

        landmarks = results.pose_landmarks.landmark
        exercise_name = request.exercise_name.lower()
        
        # --- Analyze and Get Angle/Coords ---
        if "shoulder flexion" in exercise_name or "shoulder abduction" in exercise_name:
            angle, feedback, angle_coords = analyze_shoulder_flexion(landmarks)
            MIN_ANGLE = 90
            MAX_ANGLE = 160
        elif "elbow flexion" in exercise_name or "elbow extension" in exercise_name:
            angle, feedback, angle_coords = analyze_elbow_flexion(landmarks)
            MIN_ANGLE = 60
            MAX_ANGLE = 150
        else:
            angle = 0
            feedback = [{"type": "warning", "message": "Exercise not recognized"}]
            angle_coords = {}
            MIN_ANGLE = 0
            MAX_ANGLE = 0

        # --- Rep Counting Logic (Adjusted to use MIN/MAX) ---
        # Transition to UP stage (Contraction/Lift phase)
        if angle < MIN_ANGLE + 10 and stage == "down":
            stage = "up"
            feedback.append({"type": "instruction", "message": "Hold contraction..."})

        # Transition to DOWN stage (Extension/Return phase - Rep counted)
        if angle > MAX_ANGLE - 10 and stage == "up":
            stage = "down"
            reps += 1
            feedback.append({"type": "encouragement", "message": f"Rep {reps} completed! Return slowly."})

        # --- Prepare Output Data ---
        accuracy = min(100.0, (reps / 10.0) * 100) if reps > 0 else 0.0
        
        # Extract 2D drawing data
        drawing_landmarks = get_2d_landmarks(landmarks)

        return {
            "reps": reps,
            "feedback": feedback,
            "accuracy_score": round(accuracy, 2),
            "state": {"reps": reps, "stage": stage, "angle": round(angle, 1)},
            "drawing_landmarks": drawing_landmarks, # ADDED: Skeleton drawing points
            "current_angle": round(angle, 1),       # ADDED: Current angle value
            "angle_coords": angle_coords            # ADDED: A, B, C normalized coordinates for drawing the arc
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

# ... (The /api/progress/{user_id} endpoint remains the same) ...

# =========================================================================
# 7. MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    # The separate 'main()' code block (for calibration and logging) has been removed here,
    # as it should be a separate script or integrated differently.
    uvicorn.run(app, host="0.0.0.0", port=8000)