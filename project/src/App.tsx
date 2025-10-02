import React, { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { Home } from './pages/Home';
import { LiveSession } from './pages/LiveSession';
import { Progress } from './pages/Progress';
import { Reminders } from './pages/Reminders';
import { Navigation } from './components/Navigation';

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
  difficulty_level: string;
  duration_weeks: number;
}

function AppContent() {
  const { user, loading } = useAuth();
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [currentPage, setCurrentPage] = useState('home');
  const [activeSession, setActiveSession] = useState<{
    plan: ExercisePlan;
    exercise: Exercise;
  } | null>(null);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 flex items-center justify-center">
        <div className="text-lg text-gray-600">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return authMode === 'login' ? (
      <Login onToggleMode={() => setAuthMode('register')} />
    ) : (
      <Register onToggleMode={() => setAuthMode('login')} />
    );
  }

  const handleStartSession = (plan: ExercisePlan, exercise: Exercise) => {
    setActiveSession({ plan, exercise });
  };

  const handleCompleteSession = () => {
    setActiveSession(null);
    setCurrentPage('home');
  };

  if (activeSession) {
    return (
      <LiveSession
        plan={activeSession.plan}
        exercise={activeSession.exercise}
        onComplete={handleCompleteSession}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      {currentPage === 'home' && <Home onStartSession={handleStartSession} />}
      {currentPage === 'progress' && <Progress />}
      {currentPage === 'reminders' && <Reminders />}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
