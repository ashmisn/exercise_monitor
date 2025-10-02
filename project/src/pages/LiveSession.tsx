import React, { useRef, useState, useEffect } from 'react';
import { Camera, StopCircle, Play, AlertCircle } from 'lucide-react';

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

export const LiveSession: React.FC<LiveSessionProps> = ({ plan, exercise, onComplete }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [reps, setReps] = useState(0);
  const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
  const [accuracy, setAccuracy] = useState(0);
  const [sessionState, setSessionState] = useState<any>({ reps: 0, stage: 'down' });
  const [error, setError] = useState('');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      stopSession();
    };
  }, []);

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

  const stopSession = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);

    const frameData = canvas.toDataURL('image/jpeg', 0.8);

    try {
      const response = await fetch('http://localhost:8000/api/analyze_frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frame: frameData,
          exercise_name: exercise.name,
          previous_state: sessionState,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze frame');
      }

      const data = await response.json();

      setReps(data.reps);
      setFeedback(data.feedback);
      setAccuracy(data.accuracy_score);
      setSessionState(data.state);

      if (data.reps >= exercise.target_reps) {
        stopSession();
      }
    } catch (err) {
      console.error('Analysis error:', err);
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
              onClick={onComplete}
              className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-all"
            >
              End Session
            </button>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />

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
                  onClick={stopSession}
                  className="mt-4 w-full bg-red-600 text-white py-3 rounded-lg font-medium hover:bg-red-700 transition-all inline-flex items-center justify-center"
                >
                  <StopCircle className="w-5 h-5 mr-2" />
                  Stop Session
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
                    onClick={() => {
                      setReps(0);
                      setSessionState({ reps: 0, stage: 'down' });
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
