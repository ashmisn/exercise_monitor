import base64
import cv2
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import mediapipe as mp

# =========================================================================
# 1. MEDIAPIPE & FASTAPI SETUP
# =========================================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

app = FastAPI(title="AI Physiotherapy API")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 2. DATA MODELS & CONFIGURATION
# =========================================================================
class Landmark2D(BaseModel):
    x: float
    y: float
    visibility: float = 1.0 # Assuming visibility is always 1.0 from normalized 2D list

class FrameRequest(BaseModel):
    frame: str
    exercise_name: str
    # Previous state needs to store reps, stage, and last_rep_time for debounce
    previous_state: Dict | None = None

class AilmentRequest(BaseModel): # <-- Re-defined Ailment Request Model
    ailment: str

# Exercise settings (simplified from your earlier context)
EXERCISE_CONFIGS = {
    "shoulder flexion": {
        "min_angle": 90, 
        "max_angle": 160,
        "debounce": 1.5
    },
    "elbow flexion": {
        "min_angle": 60,
        "max_angle": 150,
        "debounce": 1.5
    }
}

# Dummy Exercise Plans for get_plan endpoint (based on previous context)
EXERCISE_PLANS = {
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
# 3. UTILITY FUNCTIONS
# =========================================================================

def calculate_angle_2d(a, b, c):
    """Calculates angle between three 2D points (A-B-C) where B is the vertex."""
    a = np.array(a) 
    b = np.array(b) # Vertex point
    c = np.array(c) 

    # Ensure points are not coincident (which would cause a zero division error)
    if np.all(a == b) or np.all(c == b):
        return 0.0 # Return 0 if points overlap

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_2d_landmarks(landmarks):
    """Extracts 2D normalized landmarks and visibility for frontend drawing."""
    return [
        {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
        for lm in landmarks
    ]

# =========================================================================
# 4. EXERCISE ANALYSIS FUNCTIONS
# =========================================================================

def analyze_shoulder_flexion(landmarks):
    # Angle calculation: HIP (A) - SHOULDER (B) - ELBOW (C)
    LM_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    LM_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    LM_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    
    # Simple visibility check for required landmarks (0.5 threshold)
    if landmarks[LM_HIP].visibility < 0.5 or landmarks[LM_SHOULDER].visibility < 0.5 or landmarks[LM_ELBOW].visibility < 0.5:
        return 0, {}, [{"type": "warning", "message": "Low visibility for shoulder/hip/elbow."}]

    P_HIP = [landmarks[LM_HIP].x, landmarks[LM_HIP].y]
    P_SHOULDER = [landmarks[LM_SHOULDER].x, landmarks[LM_SHOULDER].y]
    P_ELBOW = [landmarks[LM_ELBOW].x, landmarks[LM_ELBOW].y]

    angle = calculate_angle_2d(P_HIP, P_SHOULDER, P_ELBOW)
    
    angle_coords = {
        "A": {"x": P_HIP[0], "y": P_HIP[1]},
        "B": {"x": P_SHOULDER[0], "y": P_SHOULDER[1]}, 
        "C": {"x": P_ELBOW[0], "y": P_ELBOW[1]},
    }
    return angle, angle_coords, [] # Return 0 for feedback (handled globally)

def analyze_elbow_flexion(landmarks):
    # Angle calculation: SHOULDER (A) - ELBOW (B) - WRIST (C)
    LM_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    LM_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    LM_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    
    if landmarks[LM_SHOULDER].visibility < 0.5 or landmarks[LM_ELBOW].visibility < 0.5 or landmarks[LM_WRIST].visibility < 0.5:
        return 0, {}, [{"type": "warning", "message": "Low visibility for elbow/wrist/shoulder."}]

    P_SHOULDER = [landmarks[LM_SHOULDER].x, landmarks[LM_SHOULDER].y]
    P_ELBOW = [landmarks[LM_ELBOW].x, landmarks[LM_ELBOW].y]
    P_WRIST = [landmarks[LM_WRIST].x, landmarks[LM_WRIST].y]

    angle = calculate_angle_2d(P_SHOULDER, P_ELBOW, P_WRIST)
    
    angle_coords = {
        "A": {"x": P_SHOULDER[0], "y": P_SHOULDER[1]},
        "B": {"x": P_ELBOW[0], "y": P_ELBOW[1]}, 
        "C": {"x": P_WRIST[0], "y": P_WRIST[1]},
    }
    return angle, angle_coords, [] # Return 0 for feedback (handled globally)

# Map exercise names to their analysis function
ANALYSIS_MAP = {
    "shoulder flexion": analyze_shoulder_flexion,
    "shoulder abduction": analyze_shoulder_flexion, 
    "elbow flexion": analyze_elbow_flexion,
    "elbow extension": analyze_elbow_flexion,
}

# =========================================================================
# 5. API ENDPOINTS
# =========================================================================

@app.get("/")
def root():
    return {"message": "AI Physiotherapy API is running", "status": "healthy"}

@app.post("/api/get_plan")
def get_exercise_plan(request: AilmentRequest):
    """Returns the static exercise plan for a given ailment."""
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
    # Initialize failure values
    reps, stage, last_rep_time = 0, "down", 0
    angle, angle_coords = 0, {}
    feedback = []

    # State initialization and retrieval (including debounce time)
    current_state = request.previous_state or {"reps": 0, "stage": "down", "last_rep_time": 0}
    reps = current_state.get("reps", 0)
    stage = current_state.get("stage", "down")
    last_rep_time = current_state.get("last_rep_time", 0)

    try:
        # 1. Decode Frame
        img_data = base64.b64decode(request.frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        
        # --- Handle No Pose Detected ---
        if not results.pose_landmarks:
            feedback.append({"type": "warning", "message": "No pose detected. Adjust camera view."})
        else:
            landmarks = results.pose_landmarks.landmark
            exercise_name = request.exercise_name.lower()
            
            # --- Configuration Lookup ---
            config = EXERCISE_CONFIGS.get(exercise_name, {})
            if not config:
                feedback.append({"type": "warning", "message": f"Configuration not found for: {exercise_name}"})
            else:
                # --- Analyze Angle ---
                analysis_func = ANALYSIS_MAP.get(exercise_name)
                if analysis_func:
                    angle, angle_coords, analysis_feedback = analysis_func(landmarks)
                    feedback.extend(analysis_feedback) # Add visibility warnings
                else:
                    feedback.append({"type": "warning", "message": "Analysis function missing."})

                # --- NEW: DEBUG LOGGING FOR EVERY FRAME ---
                print(f"DEBUG: Exercise={exercise_name}, ANGLE={round(angle, 1)}°, STAGE={stage}")
                # ----------------------------------------


                # --- Rep Counting Logic (with Debounce) ---
                if not analysis_feedback: # Only count if primary landmarks were visible
                    MIN_ANGLE = config['min_angle']
                    MAX_ANGLE = config['max_angle']
                    DEBOUNCE_TIME = config['debounce']
                    
                    MIN_ANGLE_THRESHOLD = MIN_ANGLE + 10
                    MAX_ANGLE_THRESHOLD = MAX_ANGLE - 10
                    current_time = time.time()
                    
                    # 1. Transition to UP stage (Contraction/Lift phase)
                    if angle < MIN_ANGLE_THRESHOLD and stage == "down":
                        stage = "up"
                        feedback.append({"type": "instruction", "message": "Hold contracted position."})

                    # 2. Transition to DOWN stage (Extension/Return phase - Rep counted)
                    if angle > MAX_ANGLE_THRESHOLD and stage == "up":
                        if current_time - last_rep_time > DEBOUNCE_TIME:
                            stage = "down"
                            reps += 1
                            last_rep_time = current_time # Update the last rep time
                            feedback.append({"type": "encouragement", "message": f"Rep {reps} completed! Return slowly."})
                            # --- SUCCESS LOGGING ---
                            print(f"SUCCESS: REPS={reps}, STAGE={stage}, ANGLE={round(angle, 1)}°")
                            # -----------------------
                        else:
                            feedback.append({"type": "warning", "message": "Too fast! Wait for the full return."})
                    
                    # If no stage transition, provide generic feedback based on angle position
                    if not any(f['type'] != 'warning' for f in feedback):
                        if angle > MAX_ANGLE_THRESHOLD:
                            feedback.append({"type": "encouragement", "message": "Ready to start your next rep."})
                        elif angle < MIN_ANGLE_THRESHOLD:
                            feedback.append({"type": "encouragement", "message": "Hold the stretch!"})
                        else:
                            feedback.append({"type": "progress", "message": "Maintain controlled movement."})
        
        # --- Prepare Output Data ---
        # If any major error occurred (like no pose), accuracy should be 0.0
        accuracy = min(100.0, (reps / 10.0) * 100) if reps > 0 and results.pose_landmarks else 0.0
        drawing_landmarks = get_2d_landmarks(landmarks) if results.pose_landmarks else []

        return {
            "reps": reps,
            "feedback": feedback if feedback else [{"type": "progress", "message": "Processing..."}],
            "accuracy_score": round(accuracy, 2),
            "state": {"reps": reps, "stage": stage, "angle": round(angle, 1), "last_rep_time": last_rep_time},
            "drawing_landmarks": drawing_landmarks,
            "current_angle": round(angle, 1),
            "angle_coords": angle_coords
        }

    except Exception as e:
        # Catch any unexpected error and log it, returning a descriptive 500
        print(f"CRITICAL ERROR in analyze_frame: {e}")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Unexpected server error during analysis: {str(e)}")

# =========================================================================
# 6. MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)