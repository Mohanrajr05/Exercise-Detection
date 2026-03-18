import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Activity, Dumbbell, Timer, Target, ArrowUpRight, ArrowDownRight, MoreHorizontal, History } from 'lucide-react';
import { motion } from 'framer-motion';

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
        average_accuracy: 0
    });
    const [loading, setLoading] = useState(true);

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
                    <button className="px-5 py-2.5 bg-zinc-800 border border-white/10 rounded-xl text-xs font-black uppercase tracking-widest text-zinc-300 hover:bg-zinc-700 transition-colors">
                        EXPORT CSV
                    </button>
                    <button className="px-5 py-2.5 bg-primary-500 text-zinc-950 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-primary-400 focus:ring-4 focus:ring-primary-500/20 transition-all shadow-[0_0_15px_rgba(132,204,22,0.3)]">
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
                        <select className="text-xs font-black uppercase tracking-widest bg-zinc-800 border border-white/10 text-zinc-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 appearance-none">
                            <option>LAST 7 DAYS</option>
                            <option>LAST 30 DAYS</option>
                            <option>THIS YEAR</option>
                        </select>
                    </div>
                    <div className="h-[300px] w-full bg-zinc-950/50 rounded-2xl border border-dashed border-white/10 flex items-center justify-center flex-col relative overflow-hidden">
                        {/* Decorative Grid */}
                        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] z-0"></div>

                        <Activity className="w-12 h-12 text-zinc-700 mb-4 relative z-10" />
                        <p className="text-[10px] font-black tracking-widest uppercase text-zinc-500 relative z-10">
                            VISUALIZATION INACTIVE
                        </p>
                    </div>
                </motion.div>

                {/* Recent Activity */}
                <motion.div variants={itemVariants} className="bg-zinc-900 p-8 rounded-3xl shadow-xl border border-white/10 flex flex-col">
                    <div className="flex justify-between items-center mb-8">
                        <h3 className="text-xl font-black text-white display-font uppercase italic">RECENT LOGS</h3>
                        <button className="text-[10px] font-black uppercase tracking-widest text-primary-500 hover:text-primary-400">VIEW ALL</button>
                    </div>

                    <div className="flex-1 flex flex-col items-center justify-center text-center p-6 bg-zinc-950/50 rounded-2xl border border-white/5 border-dashed relative overflow-hidden">
                        <History className="w-10 h-10 text-zinc-700 mb-6" />
                        <h4 className="text-zinc-300 font-bold mb-2">NO RECENT OPS</h4>
                        <p className="text-zinc-500 text-sm font-medium mb-8 leading-relaxed max-w-[200px]">Engage a training module to populate this feed.</p>
                        <button className="w-full py-3.5 bg-zinc-100 text-zinc-900 rounded-xl text-xs font-black tracking-widest uppercase hover:bg-white transition-colors">
                            INITIATE SESSION
                        </button>
                    </div>
                </motion.div>
            </motion.div>
        </motion.div>
    );
};

export default Dashboard;
