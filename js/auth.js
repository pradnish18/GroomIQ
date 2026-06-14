const API_BASE = 'http://127.0.0.1:5001';

function saveSession(token, user) {
    localStorage.setItem('groomiq_token', token);
    localStorage.setItem('groomiq_user', JSON.stringify(user));
}

function getSession() {
    const token = localStorage.getItem('groomiq_token');
    const user = localStorage.getItem('groomiq_user');
    return { token, user: user ? JSON.parse(user) : null };
}

function logout() {
    localStorage.removeItem('groomiq_token');
    localStorage.removeItem('groomiq_user');
    window.location.href = 'login.html';
}

function authHeaders() {
    const s = getSession();
    const h = { 'Content-Type': 'application/json' };
    if (s.token) h['Authorization'] = 'Bearer ' + s.token;
    return h;
}

async function signup(event) {
    event.preventDefault();
    const fullName = document.getElementById('fullName')?.value?.trim();
    const email = document.getElementById('email')?.value?.trim();
    const password = document.getElementById('password')?.value;
    try {
        const r = await fetch(`${API_BASE}/signup`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({full_name:fullName, email, password}) });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Signup failed');
        saveSession(d.token, { id: d.user_id, full_name: d.full_name, email });
        window.location.href = 'dashboard.html';
    } catch(e) { alert(e.message); }
}

async function login(event) {
    event.preventDefault();
    const email = document.getElementById('email')?.value?.trim();
    const password = document.getElementById('password')?.value;
    try {
        const r = await fetch(`${API_BASE}/login`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email, password}) });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Login failed');
        saveSession(d.token, { id: d.user_id, full_name: d.full_name, email });
        window.location.href = 'dashboard.html';
    } catch(e) { alert(e.message); }
}

async function googleAuth(credential) {
    try {
        const r = await fetch(`${API_BASE}/auth/google`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email: credential.email, full_name: credential.name, google_id: credential.sub}) });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error);
        saveSession(d.token, { id: d.user_id, full_name: d.full_name, email: credential.email });
        window.location.href = 'dashboard.html';
    } catch(e) { alert(e.message); }
}

async function githubAuth(code) {
    try {
        const r = await fetch(`https://api.github.com/user${code ? '?' + code : ''}`);
    } catch(e) { console.log(e); }
}

function requireAuth() {
    const s = getSession();
    if (!s.token) { window.location.href = 'login.html'; return false; }
    return true;
}

async function apiFetch(path, opts = {}) {
    const headers = { ...authHeaders(), ...opts.headers };
    if (opts.body && typeof opts.body === 'object' && !(opts.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
    if (res.status === 401) { logout(); throw new Error('Session expired'); }
    return res.json();
}

window.GroomIQAuth = { signup, login, logout, requireAuth, getSession, apiFetch, authHeaders, saveSession };
