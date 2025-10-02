import React, { useState } from 'react';
import { Bell, Plus, Trash2, Clock } from 'lucide-react';

interface Reminder {
  id: string;
  exercise_plan: string;
  time: string;
  days: string[];
  is_active: boolean;
}

const DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export const Reminders: React.FC = () => {
  const [reminders, setReminders] = useState<Reminder[]>([
    {
      id: '1',
      exercise_plan: 'Shoulder Injury Program',
      time: '09:00',
      days: ['Monday', 'Wednesday', 'Friday'],
      is_active: true,
    },
  ]);

  const [showAddForm, setShowAddForm] = useState(false);
  const [newReminder, setNewReminder] = useState({
    exercise_plan: 'Shoulder Injury Program',
    time: '09:00',
    days: [] as string[],
  });

  const toggleDay = (day: string) => {
    setNewReminder((prev) => ({
      ...prev,
      days: prev.days.includes(day)
        ? prev.days.filter((d) => d !== day)
        : [...prev.days, day],
    }));
  };

  const handleAddReminder = () => {
    if (newReminder.days.length === 0) {
      alert('Please select at least one day');
      return;
    }

    const reminder: Reminder = {
      id: Date.now().toString(),
      ...newReminder,
      is_active: true,
    };

    setReminders([...reminders, reminder]);
    setNewReminder({
      exercise_plan: 'Shoulder Injury Program',
      time: '09:00',
      days: [],
    });
    setShowAddForm(false);
  };

  const toggleReminder = (id: string) => {
    setReminders(
      reminders.map((r) => (r.id === id ? { ...r, is_active: !r.is_active } : r))
    );
  };

  const deleteReminder = (id: string) => {
    setReminders(reminders.filter((r) => r.id !== id));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Exercise Reminders</h1>
          <p className="text-gray-600">Set up daily reminders to stay on track with your rehabilitation</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Bell className="w-6 h-6 text-blue-600 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Active Reminders</h2>
            </div>
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-all flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Reminder
            </button>
          </div>

          {showAddForm && (
            <div className="mb-6 p-6 bg-blue-50 rounded-xl border-2 border-blue-200">
              <h3 className="text-lg font-bold text-gray-900 mb-4">New Reminder</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Exercise Plan
                  </label>
                  <select
                    value={newReminder.exercise_plan}
                    onChange={(e) => setNewReminder({ ...newReminder, exercise_plan: e.target.value })}
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option>Shoulder Injury Program</option>
                    <option>Elbow Injury Program</option>
                    <option>Wrist Injury Program</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Time
                  </label>
                  <div className="flex items-center">
                    <Clock className="w-5 h-5 text-gray-400 mr-2" />
                    <input
                      type="time"
                      value={newReminder.time}
                      onChange={(e) => setNewReminder({ ...newReminder, time: e.target.value })}
                      className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Days of Week
                  </label>
                  <div className="grid grid-cols-7 gap-2">
                    {DAYS_OF_WEEK.map((day) => (
                      <button
                        key={day}
                        onClick={() => toggleDay(day)}
                        className={`py-2 px-1 rounded-lg text-xs font-medium transition-all ${
                          newReminder.days.includes(day)
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {day.slice(0, 3)}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={handleAddReminder}
                    className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-all"
                  >
                    Save Reminder
                  </button>
                  <button
                    onClick={() => setShowAddForm(false)}
                    className="flex-1 bg-gray-100 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-200 transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-4">
            {reminders.length === 0 ? (
              <div className="text-center py-12">
                <Bell className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No reminders set</p>
                <p className="text-sm text-gray-400">Click "Add Reminder" to create your first one</p>
              </div>
            ) : (
              reminders.map((reminder) => (
                <div
                  key={reminder.id}
                  className={`border-2 rounded-xl p-6 transition-all ${
                    reminder.is_active
                      ? 'border-blue-200 bg-blue-50'
                      : 'border-gray-200 bg-gray-50 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-gray-900 mb-1">
                        {reminder.exercise_plan}
                      </h3>
                      <div className="flex items-center text-gray-600 mb-3">
                        <Clock className="w-4 h-4 mr-2" />
                        <span className="font-medium">{reminder.time}</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {reminder.days.map((day) => (
                          <span
                            key={day}
                            className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs font-medium"
                          >
                            {day}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => toggleReminder(reminder.id)}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${
                          reminder.is_active
                            ? 'bg-green-100 text-green-700 hover:bg-green-200'
                            : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                      >
                        {reminder.is_active ? 'Active' : 'Paused'}
                      </button>
                      <button
                        onClick={() => deleteReminder(reminder.id)}
                        className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-all"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-600 to-green-600 rounded-xl shadow-lg p-6 text-white">
          <h3 className="text-lg font-bold mb-2">Stay Consistent</h3>
          <p className="text-blue-100">
            Regular exercise is key to recovery. Set up reminders to maintain your rehabilitation schedule and achieve better results.
          </p>
        </div>
      </div>
    </div>
  );
};
