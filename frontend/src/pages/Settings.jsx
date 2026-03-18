import React from 'react';
import { Settings as SettingsIcon, User, Bell, Shield, Database } from 'lucide-react';
import { motion } from 'framer-motion';

const Settings = () => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-4xl mx-auto space-y-8"
        >
            <div>
                <h1 className="text-3xl font-black text-white tracking-tighter display-font uppercase italic">SYSTEM CONFIGURATION</h1>
                <p className="text-sm font-medium text-zinc-500 mt-1">Manage local state, AI tolerances, and exports.</p>
            </div>

            <div className="bg-zinc-900 rounded-3xl shadow-xl border border-white/10 overflow-hidden">
                <div className="grid grid-cols-1 md:grid-cols-4 min-h-[600px]">

                    {/* Settings Sidebar */}
                    <div className="border-r border-white/5 p-6 space-y-2 bg-zinc-950/50">
                        <button className="w-full flex items-center px-4 py-3.5 bg-primary-500/10 text-primary-500 rounded-xl text-xs font-black uppercase tracking-widest border border-primary-500/20">
                            <User className="w-5 h-5 mr-3" /> Profile
                        </button>
                        <button className="w-full flex items-center px-4 py-3.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-xl text-xs font-black uppercase tracking-widest transition-colors">
                            <SettingsIcon className="w-5 h-5 mr-3" /> AI Engine
                        </button>
                        <button className="w-full flex items-center px-4 py-3.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-xl text-xs font-black uppercase tracking-widest transition-colors">
                            <Bell className="w-5 h-5 mr-3" /> Alerts
                        </button>
                        <button className="w-full flex items-center px-4 py-3.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-xl text-xs font-black uppercase tracking-widest transition-colors">
                            <Shield className="w-5 h-5 mr-3" /> Privacy
                        </button>
                        <button className="w-full flex items-center px-4 py-3.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-xl text-xs font-black uppercase tracking-widest transition-colors">
                            <Database className="w-5 h-5 mr-3" /> Exports
                        </button>
                    </div>

                    {/* Settings Content Area */}
                    <div className="md:col-span-3 p-10">
                        <h3 className="text-xl font-black text-white mb-8 border-b border-white/10 pb-4 display-font uppercase italic">ATHLETE IDENTITY</h3>

                        <div className="space-y-8 max-w-md">
                            <div>
                                <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3">Display Name</label>
                                <input
                                    type="text"
                                    defaultValue="Mohan Raj"
                                    className="w-full px-5 py-3.5 bg-zinc-950 border border-white/10 rounded-xl text-white font-bold focus:ring-1 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all placeholder:text-zinc-600"
                                />
                            </div>

                            <div>
                                <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3">Contact Beacon</label>
                                <input
                                    type="email"
                                    defaultValue="mohan@example.com"
                                    className="w-full px-5 py-3.5 bg-zinc-950 border border-white/10 rounded-xl text-white font-bold focus:ring-1 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all placeholder:text-zinc-600"
                                />
                            </div>

                            <div>
                                <label className="block text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3">Model Accuracy Threshold</label>
                                <div className="relative">
                                    <select className="w-full px-5 py-3.5 bg-zinc-950 border border-white/10 rounded-xl text-white font-bold appearance-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all cursor-pointer">
                                        <option>STRICT (Pro Level)</option>
                                        <option>STANDARD (Enthusiast)</option>
                                        <option>LENIENT (Recreational)</option>
                                    </select>
                                    <div className="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-zinc-500">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                                        </svg>
                                    </div>
                                </div>
                                <p className="text-xs text-zinc-600 font-medium mt-3 leading-relaxed">
                                    Strict processing requires textbook perfection for successful repetition counting.
                                </p>
                            </div>

                            <div className="pt-6 border-t border-white/10">
                                <button className="px-8 py-3.5 bg-primary-500 text-zinc-900 font-black text-xs uppercase tracking-widest rounded-xl hover:bg-primary-400 transition-all shadow-[0_0_15px_rgba(132,204,22,0.2)]">
                                    COMMIT CHANGES
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default Settings;
