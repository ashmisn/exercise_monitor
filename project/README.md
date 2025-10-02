# AI Physiotherapy & Rehabilitation Application

A complete end-to-end AI-powered physiotherapy application with real-time camera-based arm movement tracking, personalized exercise plans, and progress monitoring.

## Tech Stack

- **Frontend**: React.js with TypeScript, Tailwind CSS
- **Backend**: FastAPI (Python)
- **AI/ML**: MediaPipe Pose for real-time movement tracking
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth

## Features

### User Authentication
- Secure email/password registration and login
- Session management with Supabase Auth
- Protected routes and user-specific data

### Personalized Exercise Plans
- Users enter their ailment (shoulder injury, elbow injury, wrist injury)
- System provides customized exercise plans with:
  - Exercise descriptions and instructions
  - Target repetitions and sets
  - Difficulty levels and program duration

### Live Camera Tracking
- Real-time webcam/mobile camera integration
- MediaPipe Pose AI for arm movement analysis
- Automatic repetition counting
- Live corrective feedback on posture and form
- Visual feedback overlays on camera feed

### Session Logging
- Save session data including:
  - Repetitions completed
  - Accuracy scores
  - Real-time feedback history
  - Session duration

### Progress Dashboard
- Visual charts showing weekly/monthly progress
- Track total sessions, reps, and accuracy
- Day streak tracking for consistency
- Recent session history
- Performance metrics and trends

### Exercise Reminders
- Set up daily exercise reminders
- Customizable time and days of week
- Multiple reminders per program
- Active/pause toggle for flexibility

## Installation & Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Webcam or mobile camera access

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Environment variables are already configured in `.env`:
- VITE_SUPABASE_URL
- VITE_SUPABASE_ANON_KEY

3. Start the development server:
```bash
npm run dev
```

The frontend will run at `http://localhost:5173`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Start the FastAPI server:
```bash
python main.py
```

The backend API will run at `http://localhost:8000`

You can view the interactive API documentation at `http://localhost:8000/docs`

## Running the Full Application

### For Ideathon Demo:

1. **Terminal 1** - Start Backend:
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py
```

2. **Terminal 2** - Start Frontend:
```bash
npm run dev
```

3. Open your browser to `http://localhost:5173`

4. Create an account or login

5. Select your ailment and get your exercise plan

6. Click "Start Exercise" to begin live tracking session

7. Allow camera permissions when prompted

8. Follow the on-screen instructions and watch the AI count your reps!

## API Endpoints

### GET /
Health check endpoint

### POST /api/get_plan
Get personalized exercise plan for an ailment
```json
Request: { "ailment": "shoulder injury" }
Response: {
  "ailment": "shoulder injury",
  "exercises": [...],
  "difficulty_level": "beginner",
  "duration_weeks": 6
}
```

### POST /api/analyze_frame
Analyze a single video frame for exercise tracking
```json
Request: {
  "frame": "base64_encoded_image",
  "exercise_name": "Shoulder Flexion",
  "previous_state": { "reps": 0, "stage": "down" }
}
Response: {
  "reps": 1,
  "feedback": [...],
  "accuracy_score": 85.5,
  "state": { "reps": 1, "stage": "up", "angle": 145 }
}
```

### GET /api/progress/{user_id}
Get user progress data and session history
```json
Response: {
  "total_sessions": 12,
  "total_reps": 450,
  "average_accuracy": 87.5,
  "weekly_data": [...],
  "recent_sessions": [...]
}
```

## How It Works

### AI Movement Tracking

The application uses MediaPipe Pose to detect 33 body landmarks in real-time. For arm exercises:

1. **Shoulder Flexion**: Tracks the angle between hip, shoulder, and elbow
   - Detects when arm is raised above 140°
   - Counts rep when arm returns below 80°

2. **Elbow Flexion**: Tracks the angle between shoulder, elbow, and wrist
   - Detects full flexion when angle < 60°
   - Counts rep when arm extends > 140°

3. **Real-time Feedback**: Analyzes form and provides corrections
   - "Raise your arm higher"
   - "Perfect! Full flexion achieved"
   - "Great form! Full range achieved"

### Exercise Flow

1. User logs in → Select ailment
2. System provides exercise plan
3. User starts exercise session
4. Camera activates and tracks movements
5. AI counts reps and provides feedback
6. Session data saved for progress tracking
7. User views progress dashboard

## Supported Exercises

### Shoulder Injury
- Shoulder Flexion
- Shoulder Abduction
- Shoulder Pendulum

### Elbow Injury
- Elbow Flexion
- Elbow Extension
- Wrist Rotation

### Wrist Injury
- Wrist Flexion
- Wrist Extension
- Wrist Circles

## Project Structure

```
project/
├── backend/
│   ├── main.py              # FastAPI server with MediaPipe integration
│   └── requirements.txt     # Python dependencies
├── src/
│   ├── components/
│   │   └── Navigation.tsx   # Navigation bar component
│   ├── contexts/
│   │   └── AuthContext.tsx  # Authentication context
│   ├── lib/
│   │   └── supabase.ts      # Supabase client configuration
│   ├── pages/
│   │   ├── Login.tsx        # Login page
│   │   ├── Register.tsx     # Registration page
│   │   ├── Home.tsx         # Home page with exercise plans
│   │   ├── LiveSession.tsx  # Live camera tracking session
│   │   ├── Progress.tsx     # Progress dashboard
│   │   └── Reminders.tsx    # Exercise reminders
│   ├── App.tsx              # Main application component
│   └── main.tsx             # Application entry point
└── README.md                # This file
```

## Development Notes

### Code Organization
- Modular component architecture
- Clear separation of concerns
- TypeScript for type safety
- Tailwind CSS for styling
- RESTful API design

### Camera Permissions
The application requires camera access. Make sure to:
- Allow camera permissions when prompted
- Use HTTPS in production for camera access
- Test camera functionality before demo

### Performance Tips
- Camera analysis runs every 500ms for smooth tracking
- MediaPipe Pose uses optimized models for real-time performance
- Frame compression reduces bandwidth usage

## Troubleshooting

### Camera not working
- Check browser permissions
- Ensure camera is not in use by another application
- Try refreshing the page

### Backend connection errors
- Verify backend is running at `http://localhost:8000`
- Check CORS settings if deploying to different domains
- Ensure all Python dependencies are installed

### Authentication issues
- Verify Supabase environment variables are set
- Check internet connection for Supabase Auth
- Clear browser cache and cookies

## Future Enhancements

- Voice feedback during exercises
- Multi-user support with therapist dashboard
- Exercise history export to PDF
- Integration with wearable devices
- Custom exercise creation
- Video replay of sessions
- Social features and challenges

## License

This project is created for educational and ideathon demonstration purposes.

## Support

For issues or questions, please refer to the documentation or contact the development team.
