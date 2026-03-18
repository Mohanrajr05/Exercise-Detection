import React from 'react';
import { Link } from 'react-router-dom';
import { Play, Activity, Target, Zap } from 'lucide-react';
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

const ExerciseCard = ({ title, description, badge, exerciseId, svgPath, colorClass }) => (
    <motion.div
        variants={itemVariants}
        whileHover={{ y: -5, scale: 1.02 }}
        className="relative overflow-hidden bg-zinc-900 rounded-2xl p-6 border border-white/10 shadow-xl group cursor-pointer"
    >
        {/* Glow behind card */}
        <div className={`absolute inset-0 bg-gradient-to-b ${colorClass} opacity-0 group-hover:opacity-10 transition-opacity duration-500`}></div>

        <div className="relative z-10 flex flex-col h-full">
            <div className="flex justify-between items-start mb-8">
                <span className="inline-block px-3 py-1.5 text-[10px] font-black uppercase tracking-widest rounded bg-zinc-800 text-zinc-300 border border-white/5">
                    {badge}
                </span>
                <div className={`p-2.5 rounded-xl bg-zinc-800 border border-white/5 group-hover:bg-opacity-50 transition-colors`}>
                    <svg className="w-7 h-7 text-zinc-100 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={svgPath} />
                    </svg>
                </div>
            </div>

            <h3 className="text-2xl font-black text-white mb-2 display-font uppercase italic">{title}</h3>
            <p className="text-zinc-500 text-sm mb-8 flex-1 leading-relaxed font-medium">{description}</p>

            <Link to={exerciseId ? `/workout?exercise=${exerciseId}` : "/workout"} className="group/btn relative inline-flex items-center justify-center w-full px-5 py-3 text-sm font-bold uppercase tracking-wider rounded-xl text-black bg-zinc-100 overflow-hidden">
                <div className="absolute inset-0 w-0 bg-primary-500 transition-all duration-[250ms] ease-out group-hover/btn:w-full"></div>
                <span className="relative flex items-center text-zinc-900">
                    <Play className="w-5 h-5 mr-2" fill="currentColor" />
                    Initiate Sequence
                </span>
            </Link>
        </div>
    </motion.div>
);

const MetricPill = ({ title, value, icon: Icon }) => (
    <div className="flex flex-col bg-zinc-900/80 border border-white/10 px-6 py-4 rounded-xl backdrop-blur-md">
        <div className="flex items-center text-primary-500 mb-2">
            <Icon className="w-4 h-4 mr-2" />
            <span className="text-[10px] font-black uppercase tracking-widest">{title}</span>
        </div>
        <div className="text-2xl font-black text-white display-font">{value}</div>
    </div>
);

const Home = () => {
    return (
        <motion.div
            initial="hidden"
            animate="show"
            variants={containerVariants}
            className="max-w-7xl mx-auto space-y-12 pb-12"
        >
            {/* Hero Banner Section */}
            <motion.div variants={itemVariants} className="relative bg-zinc-900 rounded-[2rem] p-10 md:p-14 overflow-hidden border border-white/10 shadow-2xl">
                {/* Aggressive Grid & Glows */}
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
                <div className="absolute -right-40 -top-40 w-96 h-96 bg-primary-500/20 blur-[100px] rounded-full pointer-events-none"></div>
                <div className="absolute right-0 bottom-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] w-1/2 h-full [mask-image:linear-gradient(to_left,white,transparent)] pointer-events-none"></div>

                <div className="relative z-10 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                    <div>
                        <div className="inline-flex items-center px-3 py-1 rounded bg-primary-500/10 border border-primary-500/30 text-primary-500 text-[10px] font-black uppercase tracking-[0.2em] mb-6">
                            <Zap className="w-3 h-3 mr-2" fill="currentColor" /> System Online
                        </div>

                        <h1 className="text-5xl md:text-7xl font-black text-white tracking-tighter mb-6 leading-[0.9] display-font uppercase italic">
                            PUSH YOUR <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-br from-primary-400 to-primary-600">LIMITS.</span>
                        </h1>

                        <p className="text-lg text-zinc-400 mb-10 max-w-xl leading-relaxed font-medium">
                            AI-powered biomechanics. Real-time form correction. Crush your goals with military-grade precision tracking directly from your webcam.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4">
                            <Link to="/workout" className="px-8 py-4 bg-primary-500 hover:bg-primary-400 text-zinc-950 rounded-xl font-black uppercase tracking-widest transition-all shadow-[0_0_20px_rgba(132,204,22,0.3)] hover:shadow-[0_0_30px_rgba(132,204,22,0.5)] hover:-translate-y-1 text-center flex items-center justify-center">
                                <Activity className="w-5 h-5 mr-2" /> Start Training
                            </Link>
                        </div>
                    </div>

                    <div className="hidden lg:flex flex-col gap-4">
                        <div className="grid grid-cols-2 gap-4">
                            <MetricPill title="AI Accuracy" value="99.4%" icon={Target} />
                            <MetricPill title="Latency" value="< 150ms" icon={Zap} />
                        </div>
                        <div className="h-48 w-full bg-zinc-800/50 rounded-xl border border-white/5 overflow-hidden relative">
                            {/* Placeholder for a cool wireframe or 3D skeleton animation if available */}
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="w-32 h-32 border-4 border-dashed border-primary-500/30 rounded-full animate-[spin_10s_linear_infinite] flex items-center justify-center">
                                    <div className="w-24 h-24 border-4 border-primary-500/50 rounded-full animate-[spin_5s_linear_infinite_reverse]"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Modules Grid */}
            <div className="px-2">
                <motion.div variants={itemVariants} className="flex justify-between items-end mb-8 border-b border-white/10 pb-4">
                    <div>
                        <h2 className="text-3xl font-black text-white tracking-tighter display-font uppercase italic">Target Modules</h2>
                    </div>
                </motion.div>

                <motion.div variants={containerVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <ExerciseCard
                        title="Push-ups"
                        description="Track elbow angle, body alignment, and count reps automatically."
                        badge="Upper Body"
                        exerciseId="pushup"
                        colorClass="from-blue-600/20 to-transparent"
                        svgPath="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                    <ExerciseCard
                        title="Squats"
                        description="Monitor knee tracking and hip depth for perfect lower body form."
                        badge="Lower Body"
                        exerciseId="squat"
                        colorClass="from-primary-600/20 to-transparent"
                        svgPath="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                    <ExerciseCard
                        title="Jumping Jacks"
                        description="Full body cardiovascular tracking evaluating arm and leg extensions."
                        badge="Cardio"
                        exerciseId="jumping_jacks"
                        colorClass="from-orange-500/20 to-transparent"
                        svgPath="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                    <ExerciseCard
                        title="Sit-ups"
                        description="Track your core contraction angles effortlessly."
                        badge="Core"
                        exerciseId="situp"
                        colorClass="from-rose-500/20 to-transparent"
                        svgPath="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <ExerciseCard
                        title="Bicep Curl"
                        description="Measure elbow flexion angles and track arm strength reps precisely."
                        badge="Upper Body"
                        exerciseId="bicep_curl"
                        colorClass="from-violet-500/20 to-transparent"
                        svgPath="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                    />
                    <ExerciseCard
                        title="Core Plank"
                        description="Real-time posture checking to ensure your back is perfectly straight."
                        badge="Core"
                        exerciseId="plank"
                        colorClass="from-emerald-600/20 to-transparent"
                        svgPath="M4 6h16M4 12h16m-7 6h7"
                    />
                    <ExerciseCard
                        title="Reverse Plank"
                        description="Hold a reverse body bridge and track posterior chain endurance."
                        badge="Core"
                        exerciseId="reverse_plank"
                        colorClass="from-cyan-500/20 to-transparent"
                        svgPath="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4"
                    />
                    <ExerciseCard
                        title="Side Plank"
                        description="Test lateral core stability with real-time body alignment feedback."
                        badge="Core"
                        exerciseId="side_plank"
                        colorClass="from-amber-500/20 to-transparent"
                        svgPath="M4 6h16M4 10h16M4 14h16M4 18h16"
                    />
                </motion.div>
            </div>
        </motion.div>
    );
};

export default Home;
