import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, History, Bell, Search, User, Mail, Shield, Save, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import api from '../../services/api';

const AVATAR_OPTIONS = [
    { seed: 'Toby&mouth=smile&eyes=default', label: 'Profile 1' },
    { seed: 'Niklas&mouth=smile&eyes=default', label: 'Profile 2' },
    { seed: 'Luis&mouth=smile&eyes=default', label: 'Profile 3' },
    { seed: 'Kimo&mouth=smile&eyes=default', label: 'Profile 4' },
    { seed: 'Aneka&mouth=smile&eyes=default', label: 'Profile 5' },
    { seed: 'Grace&mouth=smile&eyes=default', label: 'Profile 6' },
    { seed: 'Emma&mouth=smile&eyes=default', label: 'Profile 7' },
    { seed: 'Sonia&mouth=smile&eyes=default', label: 'Profile 8' }
];

const Layout = () => {
    const location = useLocation();
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [profileData, setProfileData] = useState({
        display_name: 'Athlete_01',
        email: 'athlete@fitsense.com',
        user_level: 'Intermediate',
        avatar_seed: 'Gym'
    });
    
    // Draft state for modal form to prevent "auto-save" effect in UI
    const [draftProfile, setDraftProfile] = useState(null);
    const [isChangingAvatar, setIsChangingAvatar] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    const isActive = (path) => location.pathname === path;

    useEffect(() => {
        fetchProfile();
    }, []);

    const fetchProfile = async () => {
        try {
            const response = await api.get('/api/profile/');
            setProfileData(response.data);
        } catch (error) {
            console.error("Failed to fetch profile", error);
        }
    };

    const handleOpenProfile = () => {
        setDraftProfile({ ...profileData });
        setIsChangingAvatar(false);
        setIsProfileOpen(true);
    };

    const handleCloseProfile = () => {
        setIsProfileOpen(false);
        setDraftProfile(null);
        setIsChangingAvatar(false);
    };

    const handleSaveProfile = async (e) => {
        if (e) e.preventDefault();
        setIsSaving(true);
        try {
            const response = await api.patch('/api/profile/', draftProfile);
            // Updating the main source of truth (sidebar/header) ONLY after success
            setProfileData(response.data);
            handleCloseProfile();
        } catch (error) {
            console.error("Failed to save profile", error);
            alert("Failed to save profile details.");
        } finally {
            setIsSaving(false);
        }
    };

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
                <div 
                    className="flex items-center px-6 h-20 border-b border-white/5 cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={handleOpenProfile}
                >
                    <motion.div
                        initial={{ rotate: -90, scale: 0 }}
                        animate={{ rotate: 0, scale: 1 }}
                        transition={{ type: "spring", duration: 1 }}
                        className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mr-3 shadow-[0_0_20px_rgba(132,204,22,0.4)] overflow-hidden"
                    >
                        <img 
                            src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${profileData.avatar_seed}`} 
                            alt="Logo" 
                            className="w-8 h-8 object-cover transform scale-125 translate-y-1" 
                        />
                    </motion.div>
                    <h1 className="text-2xl font-black text-white tracking-tighter display-font uppercase italic text-ellipsis overflow-hidden whitespace-nowrap">
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
                        </nav>
                    </div>
                </div>

                {/* User Footer Profile */}
                <div className="p-4 border-t border-white/5 bg-zinc-900/50">
                    <div 
                        className="flex items-center p-3 rounded-xl hover:bg-zinc-800 cursor-pointer transition-colors border border-transparent hover:border-white/5"
                        onClick={handleOpenProfile}
                    >
                        <div className="w-10 h-10 rounded-lg bg-zinc-800 flex items-center justify-center border border-zinc-700 overflow-hidden shadow-lg shadow-black/20">
                            <img 
                                src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${profileData.avatar_seed}`} 
                                alt="Avatar" 
                                className="w-full h-full object-cover transform scale-110" 
                            />
                        </div>
                        <div className="ml-3 overflow-hidden">
                            <p className="text-sm font-bold text-white uppercase tracking-wider truncate w-28">{profileData.display_name}</p>
                            <p className="text-[10px] text-primary-500 font-bold uppercase tracking-widest mt-0.5">{profileData.user_level}</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col relative w-full overflow-hidden bg-zinc-950">
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
                                className="pl-11 pr-4 py-2 bg-zinc-900/80 border border-white/10 rounded-lg text-xs font-bold uppercase tracking-wider text-zinc-300 outline-none w-64"
                            />
                        </div>
                        <button className="relative p-2 text-zinc-400 hover:text-white transition-colors">
                            <Bell className="w-5 h-5" />
                            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary-500 rounded-full"></span>
                        </button>
                    </div>
                </header>

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

            {/* Profile Modal */}
            <AnimatePresence>
                {isProfileOpen && draftProfile && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center px-4 overflow-hidden">
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={handleCloseProfile}
                            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
                        />
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="relative w-full max-w-lg bg-zinc-900 border border-white/10 rounded-3xl shadow-2xl overflow-y-auto max-h-[90vh] custom-scrollbar"
                        >
                            <div className="px-8 py-6 border-b border-white/5 flex items-center justify-between bg-zinc-900/50 sticky top-0 z-10 backdrop-blur-md">
                                <h3 className="text-xl font-black text-white italic display-font tracking-widest uppercase text-ellipsis overflow-hidden whitespace-nowrap">
                                    // Profile <span className="text-primary-500">Details</span>
                                </h3>
                                <button 
                                    onClick={handleCloseProfile}
                                    className="p-2 hover:bg-zinc-800 rounded-xl transition-colors text-zinc-500 hover:text-white shrink-0"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="p-8 space-y-6">
                                {/* Avatar Display & Selection */}
                                <div className="space-y-4">
                                    <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest ml-1 text-center">Current Avatar</label>
                                    <div className="flex flex-col items-center">
                                        <div className="w-24 h-24 rounded-2xl bg-zinc-950 border-2 border-primary-500/50 overflow-hidden shadow-xl shadow-black/50 mb-4 p-1">
                                            <img 
                                                src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${draftProfile.avatar_seed}`} 
                                                alt="Current Avatar" 
                                                className="w-full h-full object-cover transform scale-110" 
                                            />
                                        </div>
                                        
                                        {!isChangingAvatar ? (
                                            <button
                                                type="button"
                                                onClick={() => setIsChangingAvatar(true)}
                                                className="text-xs font-bold text-primary-500 hover:text-primary-400 uppercase tracking-widest py-2 px-4 rounded-lg bg-primary-500/5 hover:bg-primary-500/10 border border-primary-500/20 transition-all font-sans"
                                            >
                                                Change Avatar
                                            </button>
                                        ) : (
                                            <motion.div 
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                className="bg-zinc-950/50 border border-white/5 p-4 rounded-2xl w-full"
                                            >
                                                <div className="flex items-center justify-between mb-4 px-1">
                                                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Select New Avatar</span>
                                                    <button 
                                                        type="button"
                                                        onClick={() => setIsChangingAvatar(false)}
                                                        className="text-[10px] font-black text-zinc-500 hover:text-white uppercase"
                                                    >
                                                        Cancel
                                                    </button>
                                                </div>
                                                <div className="grid grid-cols-4 gap-2">
                                                    {AVATAR_OPTIONS.map((opt) => (
                                                        <button
                                                            key={opt.seed}
                                                            type="button"
                                                            onClick={() => {
                                                                setDraftProfile({...draftProfile, avatar_seed: opt.seed});
                                                                setIsChangingAvatar(false);
                                                            }}
                                                            className={`relative aspect-square rounded-xl overflow-hidden border-2 transition-all p-0.5 ${
                                                                draftProfile.avatar_seed === opt.seed 
                                                                ? 'border-primary-500 bg-primary-500/10' 
                                                                : 'border-zinc-800 hover:border-zinc-600'
                                                            }`}
                                                            title={opt.label}
                                                        >
                                                            <img 
                                                                src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${opt.seed}`} 
                                                                alt={opt.seed} 
                                                                className="w-full h-full object-cover transform scale-110" 
                                                            />
                                                        </button>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-2 ml-1">Display Name</label>
                                        <div className="relative">
                                            <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                                            <input
                                                type="text"
                                                value={draftProfile.display_name}
                                                onChange={(e) => setDraftProfile({...draftProfile, display_name: e.target.value})}
                                                className="w-full bg-zinc-950 border border-white/10 rounded-xl px-12 py-3 text-sm font-bold text-white focus:border-primary-500 outline-none transition-all placeholder:text-zinc-700"
                                                placeholder="ENTER NAME..."
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-2 ml-1">Contact Mail ID</label>
                                        <div className="relative">
                                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                                            <input
                                                type="email"
                                                value={draftProfile.email}
                                                onChange={(e) => setDraftProfile({...draftProfile, email: e.target.value})}
                                                className="w-full bg-zinc-950 border border-white/10 rounded-xl px-12 py-3 text-sm font-bold text-white focus:border-primary-500 outline-none transition-all placeholder:text-zinc-700"
                                                placeholder="EMAIL@EXAMPLE.COM"
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-2 ml-1">User Level</label>
                                        <div className="relative">
                                            <Shield className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                                            <select
                                                value={draftProfile.user_level}
                                                onChange={(e) => setDraftProfile({...draftProfile, user_level: e.target.value})}
                                                className="w-full bg-zinc-950 border border-white/10 rounded-xl px-12 py-3 text-sm font-bold text-white focus:border-primary-500 outline-none appearance-none transition-all cursor-pointer"
                                            >
                                                <option value="Beginner">Beginner</option>
                                                <option value="Intermediate">Intermediate</option>
                                                <option value="Pro">Pro</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 sticky bottom-0 bg-zinc-900 py-4 border-t border-white/5 -mx-8 px-8 flex gap-3">
                                    <button
                                        type="button"
                                        onClick={handleCloseProfile}
                                        className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-bold uppercase tracking-widest py-4 rounded-xl transition-all"
                                    >
                                        CANCEL
                                    </button>
                                    <button
                                        type="button"
                                        onClick={handleSaveProfile}
                                        disabled={isSaving}
                                        className="flex-[2] bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-black font-black uppercase tracking-widest py-4 rounded-xl shadow-[0_0_20px_rgba(132,204,22,0.3)] transition-all flex items-center justify-center group"
                                    >
                                        {isSaving ? (
                                            <span className="flex items-center">
                                                <Activity className="w-5 h-5 mr-3 animate-spin" />
                                                UPDATING...
                                            </span>
                                        ) : (
                                            <>
                                                <Save className="w-5 h-5 mr-3 group-hover:scale-110 transition-transform" />
                                                SAVE CHANGES
                                            </>
                                        )}
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Layout;
