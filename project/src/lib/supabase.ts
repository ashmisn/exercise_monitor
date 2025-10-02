import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface ExercisePlan {
  id: string;
  ailment: string;
  exercises: Exercise[];
  difficulty_level: string;
  duration_weeks: number;
}

export interface Exercise {
  name: string;
  description: string;
  target_reps: number;
  sets: number;
  rest_seconds: number;
}

export interface UserSession {
  id: string;
  user_id: string;
  exercise_plan_id: string;
  exercise_name: string;
  reps_completed: number;
  target_reps: number;
  accuracy_score: number;
  feedback: FeedbackItem[];
  duration_seconds: number;
  session_date: string;
}

export interface FeedbackItem {
  timestamp: number;
  message: string;
  type: 'correction' | 'encouragement' | 'warning';
}

export interface UserProgress {
  id: string;
  user_id: string;
  exercise_plan_id: string;
  total_sessions: number;
  total_reps: number;
  average_accuracy: number;
  streak_days: number;
  last_session_date: string;
}
