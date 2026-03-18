import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, History, Settings, Bell, Search, Hexagon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Layout = () => {
    const location = useLocation();
    const isActive = (path) => location.pathname === path;

    const NavItem = ({ to, icon: Icon, label }) => {
        const active = isActive(to);
        return (
            <Link
                to={to}
                className="relative flex items-center px-4 py-3 rounded-xl text-sm font-bold uppercase tracking-wider transition-all duration-300 group overflow-hidden"
            >
                {active && (
                    <motion.div
                        layoutId="active-nav"
                        className="absolute inset-0 bg-primary-500/10 border-l-4 border-primary-500 rounded-r-xl"
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                )}
                <Icon className={`w-5 h-5 mr-3 z-10 transition-colors ${active ? "text-primary-500" : "text-zinc-500 group-hover:text-zinc-300"}`} />
                <span className={`z-10 ${active ? "text-white" : "text-zinc-400 group-hover:text-zinc-200"}`}>
                    {label}
                </span>
            </Link>
        );
    };

    return (
        <div className="flex h-screen bg-zinc-950 overflow-hidden text-zinc-100 selection:bg-primary-500/30 selection:text-white font-sans">
            {/* Sidebar - Dark Aggressive Theme */}
            <aside className="w-64 bg-zinc-900/90 border-r border-white/5 flex flex-col transition-all duration-300 z-20 hidden md:flex backdrop-blur-3xl shadow-2xl">
                {/* Logo Section */}
                <div className="flex items-center px-6 h-20 border-b border-white/5">
                    <motion.div
                        initial={{ rotate: -90, scale: 0 }}
                        animate={{ rotate: 0, scale: 1 }}
                        transition={{ type: "spring", duration: 1 }}
                        className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mr-3 shadow-[0_0_20px_rgba(132,204,22,0.4)]"
                    >
                        <Activity className="w-6 h-6 text-black" fill="currentColor" strokeWidth={1} />
                    </motion.div>
                    <h1 className="text-2xl font-black text-white tracking-tighter display-font uppercase italic">
                        Fit<span className="text-primary-500">Sense</span>
                    </h1>
                </div>

                {/* Navigation Section */}
                <div className="flex-1 px-4 py-8 overflow-y-auto custom-scrollbar">
                    <div className="mb-8">
                        <p className="px-4 text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-3">Training Module</p>
                        <nav className="space-y-2">
                            <NavItem to="/" icon={LayoutDashboard} label="Command Center" />
                            <NavItem to="/workout" icon={Activity} label="Live Tracker" />
                        </nav>
                    </div>
                    <div>
                        <p className="px-4 text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-3">Performance Data</p>
                        <nav className="space-y-2">
                            <NavItem to="/dashboard" icon={History} label="Analytics" />
                            <NavItem to="/settings" icon={Settings} label="Configuration" />
                        </nav>
                    </div>
                </div>

                {/* User Footer Profile */}
                <div className="p-4 border-t border-white/5 bg-zinc-900/50">
                    <div className="flex items-center p-3 rounded-xl hover:bg-zinc-800 cursor-pointer transition-colors border border-transparent hover:border-white/5">
                        <div className="w-10 h-10 rounded-lg bg-zinc-800 flex items-center justify-center border border-zinc-700">
                            <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Gym" alt="Avatar" className="w-8 h-8 rounded-md" />
                        </div>
                        <div className="ml-3">
                            <p className="text-sm font-bold text-white uppercase tracking-wider">Athelete_01</p>
                            <p className="text-[10px] text-primary-500 font-bold uppercase tracking-widest mt-0.5">Pro Active</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col relative w-full overflow-hidden bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-zinc-950">

                {/* Background ambient lighting */}
                <div className="absolute top-[-20%] left-[-10%] w-96 h-96 bg-primary-500/10 rounded-full blur-[120px] pointer-events-none"></div>

                {/* Top Header */}
                <header className="h-20 bg-zinc-950/50 backdrop-blur-xl border-b border-white/5 flex items-center justify-between px-8 z-10 sticky top-0">

                    <div className="flex items-center">
                        <motion.h2
                            key={location.pathname}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="text-xl font-bold text-zinc-100 display-font italic tracking-wider uppercase"
                        >
                            {location.pathname === '/' ? '// OVERVIEW' : `// ${location.pathname.substring(1).replace('-', ' ')}`}
                        </motion.h2>
                    </div>

                    <div className="flex items-center space-x-6">
                        <div className="relative hidden md:block">
                            <Search className="w-4 h-4 text-zinc-500 absolute left-4 top-1/2 -translate-y-1/2" />
                            <input
                                type="text"
                                placeholder="SEARCH LOGS..."
                                className="pl-11 pr-4 py-2 bg-zinc-900/80 border border-white/10 rounded-lg text-xs font-bold uppercase tracking-wider text-zinc-300 focus:bg-zinc-900 focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500 transition-all w-64 outline-none placeholder:text-zinc-600"
                            />
                        </div>

                        <button className="relative p-2 text-zinc-400 hover:text-white transition-colors">
                            <Bell className="w-5 h-5" />
                            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary-500 rounded-full shadow-[0_0_10px_rgba(132,204,22,1)]"></span>
                        </button>
                    </div>
                </header>

                {/* Scrollable Page Content */}
                <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-8 relative z-0 custom-scrollbar">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={location.pathname}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.2 }}
                        >
                            <Outlet />
                        </motion.div>
                    </AnimatePresence>
                </div>
            </main>
        </div>
    );
};

export default Layout;
