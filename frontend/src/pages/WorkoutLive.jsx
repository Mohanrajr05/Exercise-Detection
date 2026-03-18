import React, { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import api from '../services/api';
import { Camera, Activity, Volume2, Settings2, ShieldAlert, CheckCircle2 } from 'lucide-react';

const EXERCISES = [
    { id: 'pushup', name: 'Push-ups' },
    { id: 'squat', name: 'Squats' },
    { id: 'plank', name: 'Plank' },
    { id: 'jumping_jacks', name: 'Jumping Jacks' },
    { id: 'situp', name: 'Sit-ups' }
];

const WorkoutLive = () => {
    const webcamRef = useRef(null);
    const [isTracking, setIsTracking] = useState(false);
    const [selectedExercise, setSelectedExercise] = useState('pushup');
    const [stats, setStats] = useState({ reps: 0, duration: 0, feedback: [] });
    const [fps, setFps] = useState(0);
    const requestRef = useRef();
    const lastFrameTime = useRef(performance.now());
    const frameCount = useRef(0);

    // Voice Synthesis
    const speakFeedback = useCallback((text) => {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.1;
            window.speechSynthesis.speak(utterance);
        }
    }, []);

    // Frame Capture Loop
    const captureFrame = useCallback(async () => {
        if (!isTracking || !webcamRef.current) return;

        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
            try {
                const fetchRes = await fetch(imageSrc);
                const blob = await fetchRes.blob();

                const formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');
                formData.append('exercise', selectedExercise);

                const response = await api.post('/analyze/live_frame/', formData);

                setStats(prev => {
                    const newStats = { ...prev, ...response.data };

                    const latestFeedback = response.data.feedback?.[response.data.feedback.length - 1];
                    const prevLatestFeedback = prev.feedback?.[prev.feedback.length - 1];

                    if (latestFeedback && latestFeedback !== prevLatestFeedback) {
                        speakFeedback(latestFeedback);
                    }

                    return newStats;
                });
            } catch (error) {
                console.error("Frame analysis error:", error);
            }
        }

        const now = performance.now();
        frameCount.current += 1;
        if (now - lastFrameTime.current >= 1000) {
            setFps(frameCount.current);
            frameCount.current = 0;
            lastFrameTime.current = now;
        }

        setTimeout(() => {
            if (isTracking) {
                requestRef.current = requestAnimationFrame(captureFrame);
            }
        }, 150); // Throttle

    }, [isTracking, selectedExercise, speakFeedback]);

    useEffect(() => {
        if (isTracking) {
            api.post('/analyze/reset_state/', { exercise: selectedExercise })
                .then(() => {
                    requestRef.current = requestAnimationFrame(captureFrame);
                })
                .catch(err => console.error("Could not reset backend state:", err));
        } else {
            cancelAnimationFrame(requestRef.current);
        }

        return () => cancelAnimationFrame(requestRef.current);
    }, [isTracking, captureFrame, selectedExercise]);

    const handleToggleTracking = () => {
        setIsTracking(!isTracking);
        if (!isTracking) {
            setStats({ reps: 0, duration: 0, feedback: [] });
        }
    };

    return (
        <div className="max-w-7xl mx-auto space-y-6">
            {/* Header Configuration Panel */}
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
                <div className="flex items-center">
                    <div className="w-10 h-10 bg-indigo-50 rounded-xl flex items-center justify-center mr-4 text-indigo-600">
                        <Camera className="w-5 h-5" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-slate-900 tracking-tight">Live Tracking Session</h2>
                        <p className="text-xs text-slate-500 font-medium">Configure module and press start when ready.</p>
                    </div>
                </div>

                <div className="flex items-center space-x-3 w-full md:w-auto">
                    <div className="relative flex-1 md:flex-none md:w-48">
                        <select
                            className="bg-slate-50 border border-slate-200 text-slate-700 text-sm font-semibold rounded-xl appearance-none pr-10 pl-4 py-2.5 w-full focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-shadow"
                            value={selectedExercise}
                            onChange={(e) => setSelectedExercise(e.target.value)}
                            disabled={isTracking}
                        >
                            {EXERCISES.map(ex => (
                                <option key={ex.id} value={ex.id}>{ex.name}</option>
                            ))}
                        </select>
                        <Settings2 className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                    </div>

                    <button
                        onClick={handleToggleTracking}
                        className={`px-6 py-2.5 rounded-xl font-bold text-white shadow-sm transition-all focus:ring-2 focus:ring-offset-2 ${isTracking
                                ? 'bg-rose-500 hover:bg-rose-600 focus:ring-rose-500 shadow-rose-500/30'
                                : 'bg-slate-900 hover:bg-slate-800 focus:ring-slate-900 shadow-slate-900/30'
                            }`}
                    >
                        {isTracking ? 'End Session' : 'Start Camera'}
                    </button>
                </div>
            </div>

            {/* Split Screen Application Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-220px)] min-h-[600px]">

                {/* Camera Feed Column (HUD) */}
                <div className="lg:col-span-3 relative bg-slate-900 rounded-3xl overflow-hidden border border-slate-800 shadow-xl flex items-center justify-center group">
                    <Webcam
                        ref={webcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
                        className={`w-full h-full object-cover transition-opacity duration-500 ${isTracking ? 'opacity-100' : 'opacity-40 grayscale'}`}
                    />

                    {/* HUD Overlays */}
                    {!isTracking && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10">
                            <div className="w-20 h-20 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 flex items-center justify-center mb-6">
                                <Camera className="w-8 h-8 text-white/70" />
                            </div>
                            <h3 className="text-2xl font-bold text-white tracking-tight mb-2">Camera is standing by</h3>
                            <p className="text-slate-400 max-w-sm">Position your device so your full body is visible in the frame, then begin the session.</p>
                        </div>
                    )}

                    {isTracking && (
                        <>
                            {/* Top HUD Bar */}
                            <div className="absolute top-0 inset-x-0 p-6 flex justify-between items-start bg-gradient-to-b from-black/60 to-transparent pointer-events-none">
                                <div className="flex items-center space-x-3">
                                    <div className="flex items-center space-x-2 bg-rose-500/90 backdrop-blur-md px-3 py-1.5 rounded-lg text-white text-xs font-bold shadow-lg shadow-rose-500/20">
                                        <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                                        <span className="tracking-widest uppercase">Recording</span>
                                    </div>
                                    <div className="bg-slate-900/60 backdrop-blur-md px-3 py-1.5 rounded-lg text-white text-xs font-mono border border-white/10">
                                        {fps} FPS
                                    </div>
                                </div>
                                <div className="bg-emerald-500/90 backdrop-blur-md px-3 py-1.5 rounded-lg text-white text-xs font-bold flex items-center shadow-lg shadow-emerald-500/20">
                                    <Activity className="w-3.5 h-3.5 mr-1" />
                                    Model Active
                                </div>
                            </div>

                            {/* Center Screen Flash Message (Feedback) */}
                            {stats.feedback?.length > 0 && (
                                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                                    <p className="text-5xl font-black text-white/90 drop-shadow-[0_0_15px_rgba(0,0,0,0.8)] text-center tracking-tight animate-pulse">
                                        {stats.feedback[stats.feedback.length - 1]}
                                    </p>
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Vertical Statistics Panel */}
                <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200 flex flex-col h-full overflow-hidden relative">
                    {/* Primary Metric Wrapper */}
                    <div className="text-center mb-8 relative z-10">
                        <h4 className="text-xs font-extrabold text-slate-400 uppercase tracking-widest mb-2">
                            {['plank', 'side_plank'].includes(selectedExercise) ? 'Time Active' : 'Repetitions'}
                        </h4>
                        <div className="relative inline-block">
                            <div className="text-[5rem] leading-none font-black text-slate-800 tracking-tighter tabular-nums drop-shadow-sm">
                                {['plank', 'side_plank'].includes(selectedExercise)
                                    ? `${stats.duration || 0}`
                                    : stats.reps || 0}
                            </div>
                            {['plank', 'side_plank'].includes(selectedExercise) && (
                                <span className="absolute bottom-2 -right-4 text-xl font-bold text-slate-400">s</span>
                            )}
                        </div>
                    </div>

                    {/* Timeline Action Log */}
                    <div className="flex-1 overflow-hidden flex flex-col border-t border-slate-100 pt-6">
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="font-bold text-slate-800 text-sm">Action Stream</h4>
                            <Volume2 className="w-4 h-4 text-slate-400" />
                        </div>

                        <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                            {stats.feedback?.length > 0 ? (
                                stats.feedback.map((msg, i) => {
                                    const isPositive = msg.includes("✓") || msg.toLowerCase().includes("good") || msg.toLowerCase().includes("great");
                                    return (
                                        <div key={i} className={`flex items-start p-3 rounded-xl border text-sm transition-all animate-in fade-in slide-in-from-right-2 ${isPositive
                                                ? "bg-emerald-50/50 text-emerald-800 border-emerald-100"
                                                : "bg-rose-50/50 text-rose-800 border-rose-100"
                                            }`}>
                                            {isPositive ? <CheckCircle2 className="w-4 h-4 mr-2 mt-0.5 text-emerald-500 shrink-0" /> : <ShieldAlert className="w-4 h-4 mr-2 mt-0.5 text-rose-500 shrink-0" />}
                                            <span className="font-medium leading-tight">{msg}</span>
                                        </div>
                                    );
                                }).reverse() /* Show newest at top */
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-center text-slate-400 px-4">
                                    <Activity className="w-8 h-8 mb-3 opacity-20" />
                                    <p className="text-xs font-medium uppercase tracking-wider">Awaiting Stream Data</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WorkoutLive;
