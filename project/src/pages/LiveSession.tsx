import React, { useRef, useState, useEffect } from 'react';
import { Camera, StopCircle, Play, AlertCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext'; // Original, correct import


// --- INTERFACE DEFINITIONS ---
interface Exercise {
    name: string;
    description: string;
    target_reps: number;
    sets: number;
    rest_seconds: number;
}

interface ExercisePlan {
    ailment: string;
    exercises: Exercise[];
}

interface LiveSessionProps {
    plan: ExercisePlan;
    exercise: Exercise;
    onComplete: () => void;
}

interface FeedbackItem {
    type: 'correction' | 'encouragement' | 'warning';
    message: string;
}

interface Landmark {
    x: number; 
    y: number; 
    visibility: number;
}

interface Coordinate {
    x: number;
    y: number;
}

interface DrawingData {
    landmarks: Landmark[]; 
    angleData?: { 
        angle: number, 
        A: Coordinate, 
        B: Coordinate, 
        C: Coordinate 
    }
}

const POSE_CONNECTIONS: [number, number][] = [
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];

// --- DRAWING UTILITY FUNCTION (Outside Component) ---
const drawLandmarks = (
    ctx: CanvasRenderingContext2D,
    drawingData: DrawingData,
    width: number,
    height: number,
) => {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 4;
    const { landmarks, angleData } = drawingData;

    // 1. Draw Skeleton Lines
    ctx.strokeStyle = 'rgba(76, 175, 80, 0.9)'; // Green lines
    POSE_CONNECTIONS.forEach(([i, j]) => {
        const p1 = landmarks[i];
        const p2 = landmarks[j];

        if (p1?.visibility > 0.6 && p2?.visibility > 0.6) {
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    });

    // 2. Draw Joints (Circles)
    ctx.fillStyle = 'rgb(255, 255, 255)'; // White circles
    landmarks.forEach(p => {
        if (p.visibility > 0.6) {
            ctx.beginPath();
            ctx.arc(p.x * width, p.y * height, 6, 0, 2 * Math.PI);
            ctx.fill();
        }
    });

    // 3. Draw Angle Text
    if (angleData && angleData.angle > 0) {
        const { angle, A, B, C } = angleData;

        // Convert normalized coordinates to pixel coordinates
        const center = { x: B.x * width, y: B.y * height };
        const pA = { x: A.x * width, y: A.y * height };
        const pC = { x: C.x * width, y: C.y * height };
        
        // Calculate the angle arc start and end points (simplified for drawing overlay)
        const startAngle = Math.atan2(pA.y - center.y, pA.x - center.x);
        const endAngle = Math.atan2(pC.y - center.y, pC.x - center.x);
        
        let start = startAngle < 0 ? startAngle + 2 * Math.PI : startAngle;
        let end = endAngle < 0 ? endAngle + 2 * Math.PI : endAngle;

        if (start > end) { [start, end] = [end, start]; }
        
        // Draw Angle Arc
        ctx.strokeStyle = '#FFC107'; // Amber color for arc
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(center.x, center.y, 40, start, end);
        ctx.stroke();

        // Draw Angle Text
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.font = 'bold 24px Arial';
        
        const textX = center.x + 10;
        const textY = center.y - 10;

        ctx.strokeText(`${angle.toFixed(0)}°`, textX, textY);
        ctx.fillText(`${angle.toFixed(0)}°`, textX, textY);
    }
};

// ----------------------------------------------------


export const LiveSession: React.FC<LiveSessionProps> = ({ exercise, onComplete }) => {
    const { user } = useAuth(); // Get authenticated user
    const videoRef = useRef<HTMLVideoElement>(null);
    const hiddenCanvasRef = useRef<HTMLCanvasElement>(null); 
    const drawingCanvasRef = useRef<HTMLCanvasElement>(null); 
    
    // CRITICAL FIX: Use useRef to store the state that needs to be accessed inside the interval
    const sessionStateRef = useRef<any>({ 
        reps: 0, 
        stage: 'down', 
        angle: 0, 
        last_rep_time: 0 
    }); 

    // Use useState for UI rendering only
    const [isActive, setIsActive] = useState(false);
    const [reps, setReps] = useState(0);
    const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
    const [accuracy, setAccuracy] = useState(0);
    const [error, setError] = useState('');
    const [drawingData, setDrawingData] = useState<DrawingData | null>(null);
    
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // --- NEW: Save Session Result Function ---
    const saveSessionResult = async (finalReps: number, finalAccuracy: number) => {
        // Ensure user?.id is checked for robustness, especially in a real Supabase environment
        if (!user?.id) {
            console.error("Cannot save session: User ID is missing.");
            return;
        }

        if (finalReps === 0) {
            console.log("Session ended with 0 reps, not saving.");
            return;
        }
        
        try {
            const response = await fetch('http://localhost:8000/api/save_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: user.id, // Use the real authenticated user ID
                    exercise_name: exercise.name,
                    reps_completed: finalReps,
                    accuracy_score: finalAccuracy,
                }),
            });

            if (response.ok) {
                console.log(`Successfully saved ${finalReps} reps for user ${user.id}.`);
            } else {
                const errorDetail = await response.json().catch(() => ({ detail: 'Unknown Save Error' }));
                console.error('Failed to save session:', response.status, errorDetail.detail);
            }
        } catch (error) {
            console.error('Network error while saving session:', error);
        }
    };
    // ----------------------------------------


    // --- useEffect for Drawing ---
    useEffect(() => {
        if (drawingData && drawingCanvasRef.current && isActive) {
            const canvas = drawingCanvasRef.current;
            const ctx = canvas.getContext('2d');
            const video = videoRef.current;
            
            if (ctx && video) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                drawLandmarks(
                    ctx,
                    drawingData,
                    canvas.width,
                    canvas.height
                );
            }
        } else if (drawingCanvasRef.current) {
            drawingCanvasRef.current.getContext('2d')?.clearRect(0, 0, drawingCanvasRef.current.width, drawingCanvasRef.current.height);
        }
    }, [drawingData, isActive]);
    // ----------------------------


    useEffect(() => {
        return () => {
            // Ensure session is cleaned up on component unmount
            stopSession(false); 
        };
    }, []);

    const captureAndAnalyze = async () => {
        if (!videoRef.current || !hiddenCanvasRef.current) return;

        const canvas = hiddenCanvasRef.current;
        const video = videoRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frameData = canvas.toDataURL('image/jpeg', 0.8);

        const latestState = sessionStateRef.current; 

        try {
            const response = await fetch('http://localhost:8000/api/analyze_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame: frameData,
                    exercise_name: exercise.name,
                    previous_state: latestState, 
                }),
            });

            if (!response.ok) {
                const errorDetail = await response.json().catch(() => ({ detail: 'Unknown Server Error' }));
                throw new Error(`Failed to analyze frame: ${errorDetail.detail}`);
            }

            const data = await response.json();

            // CRITICAL FIX: Update the mutable ref with the *entire* state returned by the API
            sessionStateRef.current = data.state;

            // Update local state (used for rendering UI elements like Reps/Accuracy)
            setReps(data.reps);
            setFeedback(data.feedback);
            setAccuracy(data.accuracy_score);

            // --- PROCESS DRAWING DATA ---
            if (data.drawing_landmarks && data.angle_coords) {
                setDrawingData({
                    landmarks: data.drawing_landmarks,
                    angleData: {
                        angle: data.current_angle,
                        A: data.angle_coords.A,
                        B: data.angle_coords.B,
                        C: data.angle_coords.C,
                    }
                });
            } else {
                setDrawingData(null);
            }
            // ----------------------------


            if (data.reps >= exercise.target_reps) {
                // When target reps are hit, stop session and save results.
                stopSession(true); 
            }
        } catch (err) {
            console.error('Analysis error:', err);
            setError('Connection or Analysis Error. Check backend console.');
            setDrawingData(null);
        }
    };

    // Modified stopSession to accept a boolean flag for saving data
    const stopSession = (shouldSave: boolean) => {
        // Stop the interval
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        // Stop camera tracks
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
        }

        if (shouldSave) {
            // Save the final result before resetting the state
            saveSessionResult(sessionStateRef.current.reps, accuracy);
        }
        
        // Reset state ref to initial values when stopping
        sessionStateRef.current = { reps: 0, stage: 'down', angle: 0, last_rep_time: 0 };
        setDrawingData(null); 
        setIsActive(false);
    };

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }

            setIsActive(true);
            setError('');

            intervalRef.current = setInterval(() => {
                captureAndAnalyze();
            }, 500); 
        } catch (err) {
            setError('Failed to access camera. Please grant camera permissions.');
        }
    };

    const getFeedbackColor = (type: string) => {
        switch (type) {
            case 'correction':
                return 'bg-yellow-50 border-yellow-200 text-yellow-800';
            case 'encouragement':
                return 'bg-green-50 border-green-200 text-green-800';
            case 'warning':
                return 'bg-red-50 border-red-200 text-red-800';
            default:
                return 'bg-gray-50 border-gray-200 text-gray-800';
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 py-8">
            <div className="max-w-7xl mx-auto px-4">
                <div className="bg-white rounded-2xl shadow-xl p-8">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900 mb-2">
                                {exercise.name}
                            </h1>
                            <p className="text-gray-600">{exercise.description}</p>
                        </div>
                        <button
                            // Calls stopSession but does NOT save, then calls onComplete to leave page
                            onClick={() => {
                                stopSession(false);
                                onComplete();
                            }} 
                            className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-all"
                        >
                            End Session (Don't Save)
                        </button>
                    </div>

                    <div className="grid lg:grid-cols-2 gap-8">
                        <div>
                            <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video">
                                {/* Video element shows the live stream */}
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full h-full object-cover"
                                />
                                
                                {/* Hidden canvas for capturing frames to send to API */}
                                <canvas ref={hiddenCanvasRef} className="hidden" />

                                {/* Drawing canvas OVERLAY for skeleton and angles */}
                                <canvas
                                    ref={drawingCanvasRef}
                                    className="absolute top-0 left-0 w-full h-full object-cover"
                                />

                                {/* Start/Stop UI overlay */}
                                {!isActive && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
                                        <div className="text-center">
                                            <Camera className="w-16 h-16 text-white mx-auto mb-4" />
                                            <p className="text-white text-lg mb-4">Camera not active</p>
                                            <button
                                                onClick={startCamera}
                                                className="bg-green-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-green-700 transition-all inline-flex items-center"
                                            >
                                                <Play className="w-5 h-5 mr-2" />
                                                Start Camera
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {error && (
                                <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-center">
                                    <AlertCircle className="w-5 h-5 mr-2" />
                                    {error}
                                </div>
                            )}

                            {isActive && (
                                <button
                                    // Stops the session and saves the results
                                    onClick={() => stopSession(true)}
                                    className="mt-4 w-full bg-red-600 text-white py-3 rounded-lg font-medium hover:bg-red-700 transition-all inline-flex items-center justify-center"
                                >
                                    <StopCircle className="w-5 h-5 mr-2" />
                                    Stop Session & Save
                                </button>
                            )}
                        </div>

                        <div className="space-y-6">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-blue-50 rounded-xl p-6">
                                    <div className="text-4xl font-bold text-blue-600 mb-2">
                                        {reps}
                                    </div>
                                    <div className="text-sm text-gray-600">
                                        Reps Completed
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        Target: {exercise.target_reps}
                                    </div>
                                </div>

                                <div className="bg-green-50 rounded-xl p-6">
                                    <div className="text-4xl font-bold text-green-600 mb-2">
                                        {accuracy.toFixed(0)}%
                                    </div>
                                    <div className="text-sm text-gray-600">
                                        Accuracy Score
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gray-50 rounded-xl p-6">
                                <h3 className="text-lg font-bold text-gray-900 mb-4">
                                    Exercise Details
                                </h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Target Reps:</span>
                                        <span className="font-medium">{exercise.target_reps}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Sets:</span>
                                        <span className="font-medium">{exercise.sets}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Rest Time:</span>
                                        <span className="font-medium">{exercise.rest_seconds}s</span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-white border-2 border-gray-200 rounded-xl p-6">
                                <h3 className="text-lg font-bold text-gray-900 mb-4">
                                    Real-Time Feedback
                                </h3>
                                <div className="space-y-3">
                                    {feedback.length > 0 ? (
                                        feedback.map((item, index) => (
                                            <div
                                                key={index}
                                                className={`px-4 py-3 rounded-lg border ${getFeedbackColor(item.type)}`}
                                            >
                                                {item.message}
                                            </div>
                                        ))
                                    ) : (
                                        <div className="text-gray-500 text-sm text-center py-4">
                                            {isActive ? 'Position yourself in front of the camera...' : 'Start the session to receive feedback'}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {reps >= exercise.target_reps && (
                                <div className="bg-green-50 border-2 border-green-200 rounded-xl p-6 text-center">
                                    <div className="text-2xl font-bold text-green-800 mb-2">
                                        Set Complete!
                                    </div>
                                    <p className="text-green-700 mb-4">
                                        Great job! Take a {exercise.rest_seconds} second rest.
                                    </p>
                                    <button
                                        // This button means the user acknowledges the rest and starts the next set, 
                                        // but the results of the *completed* set should have already been saved by stopSession(true)
                                        onClick={() => {
                                            setReps(0);
                                            // Reset the ref state to initial values
                                            sessionStateRef.current = { reps: 0, stage: 'down', angle: 0, last_rep_time: 0 };
                                        }}
                                        className="bg-green-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-green-700 transition-all"
                                    >
                                        Start Next Set
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};