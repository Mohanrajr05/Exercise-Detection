import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Activity, Dumbbell, Timer, Target, ArrowUpRight, ArrowDownRight, MoreHorizontal, History, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const containerVariants = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: { staggerChildren: 0.1 }
    }
};

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const StatCard = ({ title, value, subtitle, trend, trendValue, icon: Icon, colorClass }) => (
    <motion.div
        variants={itemVariants}
        className="bg-zinc-900 p-6 rounded-2xl shadow-xl border border-white/10 relative overflow-hidden group"
    >
        <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full ${colorClass.replace('text', 'bg')} opacity-5 blur-2xl group-hover:scale-150 transition-transform duration-700`}></div>

        <div className="flex justify-between items-start mb-4 relative z-10">
            <div className={`p-3 rounded-xl bg-zinc-800 border border-white/5 ${colorClass}`}>
                <Icon className="w-6 h-6" />
            </div>
            <button className="text-zinc-500 hover:text-zinc-300 transition-colors">
                <MoreHorizontal className="w-5 h-5" />
            </button>
        </div>

        <div className="relative z-10">
            <h3 className="text-4xl font-black text-white tracking-tighter display-font">{value}</h3>
            <p className="text-xs font-bold uppercase tracking-widest text-zinc-500 mt-1">{title}</p>
        </div>

        {(trend && trendValue) && (
            <div className="flex items-center mt-6 relative z-10 pt-4 border-t border-white/5">
                <div className={`flex items-center text-xs font-black px-2 py-1 rounded bg-zinc-800 border border-white/5 ${trend === 'up' ? 'text-primary-500' : 'text-rose-500'
                    }`}>
                    {trend === 'up' ? <ArrowUpRight className="w-3 h-3 mr-1" /> : <ArrowDownRight className="w-3 h-3 mr-1" />}
                    {trendValue}
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-600 ml-3">{subtitle}</span>
            </div>
        )}
    </motion.div>
);

const Dashboard = () => {
    const [stats, setStats] = useState({
        total_workouts: 0,
        total_reps: 0,
        total_duration: 0,
        average_accuracy: 0,
        recent_sessions: []
    });
    const [loading, setLoading] = useState(true);
    
    // Volume Trend State
    const [volumeData, setVolumeData] = useState([]);
    const [timeRange, setTimeRange] = useState('7'); // '7', '30', '365'
    
    // Export Modal State
    const [isExportModalOpen, setIsExportModalOpen] = useState(false);
    const [exportType, setExportType] = useState('pdf'); // 'pdf' or 'csv'
    const [exportMode, setExportMode] = useState('single'); // 'single' or 'range'
    const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                setLoading(true);
                const response = await api.get('/api/analytics/summary/');
                setStats(response.data);
            } catch (err) {
                console.error("Failed to load analytics", err);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    useEffect(() => {
        const fetchVolume = async () => {
            try {
                const response = await api.get(`/api/analytics/volume/?days=${timeRange}`);
                setVolumeData(response.data);
            } catch (err) {
                console.error("Failed to load volume data", err);
            }
        };
        fetchVolume();
    }, [timeRange]);

    const handleDownload = async () => {
        try {
            const url = `/api/analytics/export/${exportType}/?type=${exportMode}&start=${startDate}&end=${endDate}`;
            const response = await api.get(url, { responseType: 'blob' });
            
            // Check if the response is actually JSON (an error) instead of the requested blob
            if (response.data.type === 'application/json') {
                const text = await response.data.text();
                const errorData = JSON.parse(text);
                alert(`Export Error: ${errorData.error || 'Unknown error'}`);
                return;
            }

            const blob = new Blob([response.data], { 
                type: exportType === 'csv' ? 'text/csv' : 'application/pdf' 
            });
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.setAttribute('download', exportType === 'csv' ? `workout_data_${startDate}.csv` : `workout_report_${startDate}.pdf`);
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);
            setIsExportModalOpen(false);
        } catch (error) {
            console.error("Export failed", error);
            alert("Export failed. Please check the console for details.");
        }
    };

    if (loading) return (
        <div className="flex h-[80vh] items-center justify-center">
            <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full border-t-4 border-primary-500 animate-spin"></div>
                <div className="absolute inset-2 rounded-full border-r-4 border-primary-700 animate-[spin_1.5s_linear_infinite_reverse]"></div>
            </div>
        </div>
    );

    return (
        <motion.div
            initial="hidden"
            animate="show"
            variants={containerVariants}
            className="space-y-8 max-w-7xl mx-auto"
        >
            {/* Header Actions */}
            <motion.div variants={itemVariants} className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-black text-white tracking-tighter display-font uppercase italic">PERFORMANCE METRICS</h1>
                    <p className="text-sm font-medium text-zinc-500 mt-1">Aggregated tracking data across all modules.</p>
                </div>
                <div className="flex space-x-3">
                    <button 
                        onClick={() => { setExportType('csv'); setIsExportModalOpen(true); }}
                        className="px-5 py-2.5 bg-zinc-800 border border-white/10 rounded-xl text-xs font-black uppercase tracking-widest text-zinc-300 hover:bg-zinc-700 transition-colors">
                        EXPORT CSV
                    </button>
                    <button 
                        onClick={() => { setExportType('pdf'); setIsExportModalOpen(true); }}
                        className="px-5 py-2.5 bg-primary-500 text-zinc-950 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-primary-400 focus:ring-4 focus:ring-primary-500/20 transition-all shadow-[0_0_15px_rgba(132,204,22,0.3)]">
                        GENERATE REPORT
                    </button>
                </div>
            </motion.div>

            {/* KPI Cards */}
            <motion.div variants={containerVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    title="Total Sessions"
                    value={stats.total_workouts}
                    subtitle="VS LAST MONTH"
                    trend="up"
                    trendValue="12%"
                    colorClass="text-indigo-400"
                    icon={Activity}
                />
                <StatCard
                    title="Total Reps"
                    value={stats.total_reps}
                    subtitle="VS LAST MONTH"
                    trend="up"
                    trendValue="8%"
                    colorClass="text-emerald-400"
                    icon={Dumbbell}
                />
                <StatCard
                    title="Time Active"
                    value={`${Math.round(stats.total_duration / 60)}m`}
                    subtitle="VS LAST MONTH"
                    trend="down"
                    trendValue="2%"
                    colorClass="text-amber-400"
                    icon={Timer}
                />
                <StatCard
                    title="Avg Accuracy"
                    value={`${stats.average_accuracy}%`}
                    subtitle="LIFETIME RATING"
                    trend="up"
                    trendValue="4.2%"
                    colorClass="text-primary-500"
                    icon={Target}
                />
            </motion.div>

            {/* Main Charts & Tables Area */}
            <motion.div variants={containerVariants} className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Main Chart Placeholder */}
                <motion.div variants={itemVariants} className="lg:col-span-2 bg-zinc-900 p-8 rounded-3xl shadow-xl border border-white/10">
                    <div className="flex justify-between items-center mb-8">
                        <h3 className="text-xl font-black text-white display-font uppercase italic">VOLUME TREND</h3>
                        <select 
                            className="text-xs font-black uppercase tracking-widest bg-zinc-800 border border-white/10 text-zinc-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 appearance-none"
                            value={timeRange}
                            onChange={(e) => setTimeRange(e.target.value)}
                        >
                            <option value="7">LAST 7 DAYS</option>
                            <option value="30">LAST 30 DAYS</option>
                            <option value="365">THIS YEAR</option>
                        </select>
                    </div>
                    <div className="h-[300px] w-full bg-zinc-950/50 rounded-2xl border border-dashed border-white/10 relative overflow-hidden p-4">
                        {volumeData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={volumeData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                    <XAxis dataKey="date" stroke="#666" tick={{fill: '#666', fontSize: 10}} tickMargin={10} />
                                    <YAxis yAxisId="left" stroke="#84cc16" tick={{fill: '#666', fontSize: 10}} />
                                    <YAxis yAxisId="right" orientation="right" stroke="#60a5fa" tick={{fill: '#666', fontSize: 10}} />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#333', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff', fontSize: '12px', fontWeight: 'bold' }}
                                    />
                                    <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                                    <Bar yAxisId="left" dataKey="reps" name="Reps Volume" fill="#84cc16" radius={[4,4,0,0]} barSize={20} />
                                    <Line yAxisId="right" type="monotone" dataKey="duration" name="Duration (s)" stroke="#60a5fa" strokeWidth={3} dot={{r: 4, fill: '#18181b', strokeWidth: 2}} />
                                </ComposedChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <Activity className="w-12 h-12 text-zinc-700 mb-4" />
                                <p className="text-[10px] font-black tracking-widest uppercase text-zinc-500">
                                    AWAITING DATA
                                </p>
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Recent Activity */}
                <motion.div variants={itemVariants} className="bg-zinc-900 p-8 rounded-3xl shadow-xl border border-white/10 flex flex-col">
                    <div className="flex justify-between items-center mb-8">
                        <h3 className="text-xl font-black text-white display-font uppercase italic">RECENT LOGS</h3>
                        <button className="text-[10px] font-black uppercase tracking-widest text-primary-500 hover:text-primary-400">VIEW ALL</button>
                    </div>

                    {stats.recent_sessions && stats.recent_sessions.length > 0 ? (
                        <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar mt-4">
                            {stats.recent_sessions.map((session, i) => (
                                <div key={i} className="bg-zinc-950/50 p-4 rounded-xl border border-white/5 flex items-center justify-between group hover:border-white/10 transition-colors">
                                    <div className="flex items-center space-x-4">
                                        <div className="w-10 h-10 rounded-lg bg-zinc-900 border border-white/5 flex items-center justify-center text-primary-500">
                                            <Activity className="w-5 h-5" />
                                        </div>
                                        <div className="text-left">
                                            <h5 className="text-white font-bold capitalize">{session.exercise_type.replace('_', ' ')}</h5>
                                            <p className="text-xs text-zinc-500 font-medium">
                                                {new Date(session.date).toLocaleDateString()} at {new Date(session.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-lg font-black text-white tabular-nums">
                                            {session.reps > 0 ? session.reps : Math.round(session.duration_seconds)}
                                        </div>
                                        <div className="text-[10px] font-bold tracking-widest uppercase text-zinc-500">
                                            {session.reps > 0 ? 'REPS' : 'SEC'}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 bg-zinc-950/50 rounded-2xl border border-white/5 border-dashed relative overflow-hidden">
                            <History className="w-10 h-10 text-zinc-700 mb-6" />
                            <h4 className="text-zinc-300 font-bold mb-2">NO RECENT OPS</h4>
                            <p className="text-zinc-500 text-sm font-medium mb-8 leading-relaxed max-w-[200px]">Engage a training module to populate this feed.</p>
                            <button className="w-full py-3.5 bg-zinc-100 text-zinc-900 rounded-xl text-xs font-black tracking-widest uppercase hover:bg-white transition-colors">
                                INITIATE SESSION
                            </button>
                        </div>
                    )}
                </motion.div>
            </motion.div>

            {/* Export Modal */}
            <AnimatePresence>
                {isExportModalOpen && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
                        <motion.div 
                            initial={{ opacity: 0 }} 
                            animate={{ opacity: 1 }} 
                            exit={{ opacity: 0 }} 
                            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
                            onClick={() => setIsExportModalOpen(false)}
                        />
                        <motion.div 
                            initial={{ scale: 0.95, opacity: 0, y: 20 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.95, opacity: 0, y: 20 }}
                            className="bg-zinc-900 border border-white/10 p-8 rounded-3xl shadow-2xl relative z-10 w-full max-w-md"
                        >
                            <button 
                                onClick={() => setIsExportModalOpen(false)}
                                className="absolute top-6 right-6 text-zinc-500 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                            
                            <h2 className="text-2xl font-black text-white display-font uppercase italic mb-2">
                                {exportType === 'csv' ? 'Export Data' : 'Generate Report'}
                            </h2>
                            <p className="text-sm font-medium text-zinc-400 mb-8">
                                Select the timeframe for your archive extraction.
                            </p>
                            
                            <div className="space-y-6">
                                <div className="grid grid-cols-2 gap-3 p-1 bg-zinc-950 rounded-xl">
                                    <button 
                                        onClick={() => setExportMode('single')}
                                        className={`py-2 px-4 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${exportMode === 'single' ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                                    >
                                        Single Day
                                    </button>
                                    <button 
                                        onClick={() => setExportMode('range')}
                                        className={`py-2 px-4 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${exportMode === 'range' ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                                    >
                                        Date Range
                                    </button>
                                </div>
                                
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-[10px] font-black uppercase tracking-widest text-zinc-500 mb-2">
                                            {exportMode === 'single' ? 'Target Date' : 'Start Date'}
                                        </label>
                                        <input 
                                            type="date" 
                                            value={startDate}
                                            onChange={(e) => setStartDate(e.target.value)}
                                            className="w-full bg-zinc-950 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                                        />
                                    </div>
                                    
                                    {exportMode === 'range' && (
                                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                                            <label className="block text-[10px] font-black uppercase tracking-widest text-zinc-500 mb-2">
                                                End Date
                                            </label>
                                            <input 
                                                type="date" 
                                                value={endDate}
                                                onChange={(e) => setEndDate(e.target.value)}
                                                className="w-full bg-zinc-950 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all"
                                            />
                                        </motion.div>
                                    )}
                                </div>
                                
                                <button 
                                    onClick={handleDownload}
                                    className="w-full py-4 bg-primary-500 text-zinc-950 rounded-xl text-sm font-black uppercase tracking-widest hover:bg-primary-400 focus:ring-4 focus:ring-primary-500/20 transition-all mt-4"
                                >
                                    Confirm Download
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default Dashboard;
