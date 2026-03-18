import axios from 'axios';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000', // Set VITE_API_URL in Vercel for production
});

export default api;
