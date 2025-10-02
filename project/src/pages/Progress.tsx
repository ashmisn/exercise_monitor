import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { TrendingUp, Calendar, Target, Award, Activity } from 'lucide-react';

interface ProgressData {
  user_id: string;
  total_sessions: number;
  total_reps: number;
  average_accuracy: number;
  streak_days: number;
  weekly_data: Array<{
    day: string;
    reps: number;
    accuracy: number;
  }>;
  recent_sessions: Array<{
    date: string;
    exercise: string;
    reps: number;
    accuracy: number;
  }>;
}

export const Progress: React.FC = () => {
  const { user } = useAuth();
  const [progressData, setProgressData] = useState<ProgressData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchProgress();
    }
  }, [user]);

  const fetchProgress = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/progress/${user?.id}`);
      const data = await response.json();
      setProgressData(data);
    } catch (err) {
      console.error('Failed to fetch progress:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 flex items-center justify-center">
        <div className="text-lg text-gray-600">Loading progress...</div>
      </div>
    );
  }

  if (!progressData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 flex items-center justify-center">
        <div className="text-lg text-gray-600">No progress data available</div>
      </div>
    );
  }

  const maxReps = Math.max(...progressData.weekly_data.map((d) => d.reps));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Your Progress</h1>
          <p className="text-gray-600">Track your rehabilitation journey</p>
        </div>

        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-blue-100 p-3 rounded-lg">
                <Activity className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {progressData.total_sessions}
            </div>
            <div className="text-sm text-gray-600">Total Sessions</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-green-100 p-3 rounded-lg">
                <Target className="w-6 h-6 text-green-600" />
              </div>
            </div>
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {progressData.total_reps}
            </div>
            <div className="text-sm text-gray-600">Total Reps</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-yellow-100 p-3 rounded-lg">
                <TrendingUp className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {progressData.average_accuracy.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Avg Accuracy</div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-orange-100 p-3 rounded-lg">
                <Award className="w-6 h-6 text-orange-600" />
              </div>
            </div>
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {progressData.streak_days}
            </div>
            <div className="text-sm text-gray-600">Day Streak</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Calendar className="w-6 h-6 text-blue-600 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Weekly Activity</h2>
            </div>

            <div className="space-y-4">
              {progressData.weekly_data.map((day, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">{day.day}</span>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-gray-600">{day.reps} reps</span>
                      <span className="text-green-600 font-medium">{day.accuracy}%</span>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-green-500 h-full rounded-full transition-all"
                      style={{ width: `${(day.reps / maxReps) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-6 border-t border-gray-200">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Weekly Average</span>
                <span className="font-semibold text-gray-900">
                  {(progressData.weekly_data.reduce((sum, d) => sum + d.reps, 0) / 7).toFixed(0)} reps/day
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Activity className="w-6 h-6 text-green-600 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Recent Sessions</h2>
            </div>

            <div className="space-y-4">
              {progressData.recent_sessions.map((session, index) => (
                <div
                  key={index}
                  className="border-2 border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-all"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-semibold text-gray-900">{session.exercise}</h3>
                      <p className="text-sm text-gray-600">
                        {new Date(session.date).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-blue-600">{session.reps}</div>
                      <div className="text-xs text-gray-600">reps</div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-green-500 h-full rounded-full"
                        style={{ width: `${session.accuracy}%` }}
                      />
                    </div>
                    <span className="ml-3 text-sm font-medium text-green-600">
                      {session.accuracy}%
                    </span>
                  </div>
                </div>
              ))}
            </div>

            <button className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-all">
              View All Sessions
            </button>
          </div>
        </div>

        <div className="mt-8 bg-gradient-to-r from-blue-600 to-green-600 rounded-xl shadow-lg p-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold mb-2">Keep up the great work!</h2>
              <p className="text-blue-100">
                You're making excellent progress on your rehabilitation journey.
                {progressData.streak_days >= 5 && ' Your consistency is impressive!'}
              </p>
            </div>
            <div className="bg-white bg-opacity-20 backdrop-blur-sm rounded-xl p-6 text-center">
              <div className="text-4xl font-bold mb-1">{progressData.streak_days}</div>
              <div className="text-sm text-blue-100">Day Streak</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
