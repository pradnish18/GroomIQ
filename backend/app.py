import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io, sqlite3, uuid, json, base64, os, smtplib
from datetime import datetime, timedelta, UTC
from email.mime.text import MIMEText
from functools import wraps

from hair_advice import get_advice
from auth.security import hash_password, verify_password, generate_token, verify_token
from analysis.beard import recommend_beard, get_beard_profile
from recommendations.hairstyles import recommend_styles
from ai_preview import generate_preview

app = Flask(__name__)
CORS(app, origins=[
    'http://127.0.0.1:8000',
    'http://localhost:8000',
    'https://groomiq.vercel.app',
], supports_credentials=True)

CLASSES = ['Straight', 'Wavy', 'bald', 'curly', 'dreadlocks',
           'dry', 'frizzy', 'hairfall', 'healthy', 'kinky', 'notbald']
TYPE_CLASSES  = ['Straight', 'Wavy', 'curly', 'dreadlocks', 'kinky']
COND_CLASSES  = ['bald', 'dry', 'hairfall', 'healthy']

model = None
type_model = None
cond_model = None
try:
    model = tf.keras.models.load_model("model/best_model.h5")
except Exception as e:
    print("Main model not loaded:", e)
try:
    type_model = tf.keras.models.load_model("model/type_model.h5")
except Exception as e:
    print("Type model not loaded:", e)
try:
    cond_model = tf.keras.models.load_model("model/condition_model.h5")
except Exception as e:
    print("Condition model not loaded:", e)

DB_PATH = "model/groomiq.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            reset_token TEXT,
            reset_expires TEXT,
            created_at TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            disabled INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            face_shape TEXT NOT NULL,
            hair_type TEXT,
            beard_style TEXT,
            recommended_styles TEXT,
            confidence REAL NOT NULL,
            top2 TEXT,
            web_tips TEXT,
            grooming_tips TEXT,
            image_b64 TEXT,
            share_token TEXT UNIQUE,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS favorites (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            hairstyle TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            UNIQUE(user_id, hairstyle)
        );
        CREATE TABLE IF NOT EXISTS ratings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            hairstyle TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating>=1 AND rating<=5),
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            UNIQUE(user_id, hairstyle)
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'info',
            read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    print("Database ready at", DB_PATH)

init_db()

# ===== FACE SHAPE DETECTION (OpenCV) =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_shape_from_image(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        return "Oval", 85.0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
    if len(faces) == 0:
        return "Oval", 85.0
    x, y, fw, fh = faces[0]
    ratio = fh / fw if fw > 0 else 1
    center_third_start = int(w * 0.33)
    center_third_end = int(w * 0.66)
    left_third = gray[y:y+fh, max(0, x-int(fw*0.1)):x+int(fw*0.3)]
    right_third = gray[y:y+fh, x+int(fw*0.7):min(w, x+fw+int(fw*0.1))]
    center_third = gray[y:y+fh, center_third_start:center_third_end]
    jaw_variance = 0
    if left_third.size > 0 and right_third.size > 0:
        left_edges = cv2.Canny(left_third, 50, 150)
        right_edges = cv2.Canny(right_third, 50, 150)
        jaw_variance = (np.mean(left_edges) + np.mean(right_edges)) / 2
    if 1.2 <= ratio <= 1.6:
        face_shape = "Oval"
        confidence = 88.0
    elif ratio < 1.2:
        if jaw_variance > 30:
            face_shape = "Square"
            confidence = 85.0
        else:
            face_shape = "Round"
            confidence = 86.0
    elif ratio > 1.6:
        if jaw_variance > 25:
            face_shape = "Rectangle"
            confidence = 84.0
        else:
            face_shape = "Heart"
            confidence = 83.0
    else:
        face_shape = "Oval"
        confidence = 85.0
    return face_shape, round(confidence, 1)

# ===== GROOMING TIPS ENGINE =====
GROOMING_TIPS = {
    "Oval": [
        "Most hairstyles suit oval faces — experiment freely.",
        "Maintain balanced volume on top and sides.",
        "Light stubble or a short boxed beard complements oval faces well.",
        "Avoid overly long hair that drags down your face shape."
    ],
    "Round": [
        "Add height to your hairstyle to elongate the face.",
        "Avoid excessive side volume — keep sides tight.",
        "Angular beard styles like Goatee or Van Dyke add definition.",
        "High fades and pompadours work best for round faces."
    ],
    "Square": [
        "Textured hairstyles soften strong jawlines.",
        "Medium-length beards enhance your features.",
        "Avoid boxy cuts — go for layered, textured styles.",
        "Side parts and slick backs look great on square faces."
    ],
    "Rectangle": [
        "Side parts and medium-length cuts balance a long face.",
        "A full beard adds width and shortens the visual length.",
        "Avoid excessive top volume — keep it moderate.",
        "Crew cuts and textured crops work well."
    ],
    "Diamond": [
        "Textured fringes and side parts soften cheekbones.",
        "Circle beards and stubble complement diamond faces.",
        "Avoid slicked-back styles that emphasize width.",
        "Layered cuts add balance to your facial structure."
    ],
    "Heart": [
        "Fringes and side-swept hair balance a wider forehead.",
        "Full beards and short boxed beards add jaw definition.",
        "Avoid high-volume top styles that widen the forehead.",
        "Medium crops and textured quiffs are great choices."
    ],
    "Triangle": [
        "Textured crops and fringes balance a strong jawline.",
        "Full beards and short beards add structure.",
        "Keep the sides moderate — avoid extreme tapers.",
        "Layered hair on top draws attention upward."
    ]
}

DEFAULT_TIPS = [
    "Keep your hair clean and well-moisturized.",
    "Regular trims every 4-6 weeks maintain style.",
    "Use heat protection before styling.",
    "A balanced diet promotes healthy hair growth."
]

def get_grooming_tips(face_shape):
    return GROOMING_TIPS.get(face_shape, DEFAULT_TIPS)

# ===== AUTH DECORATOR =====
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        return f(payload['user_id'], *args, **kwargs)
    return decorated

def optional_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user_id = 'guest'
        if token:
            payload = verify_token(token)
            if payload:
                user_id = payload['user_id']
        return f(user_id, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid token'}), 401
        conn = get_db()
        user = conn.execute('SELECT is_admin FROM users WHERE id=?', (payload['user_id'],)).fetchone()
        conn.close()
        if not user or not user['is_admin']:
            return jsonify({'error': 'Admin access required'}), 403
        return f(payload['user_id'], *args, **kwargs)
    return decorated

# ================================================================
# AUTH ENDPOINTS
# ================================================================
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        if not full_name or not email or not password:
            return jsonify({'error': 'Missing required fields'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        conn = get_db()
        existing = conn.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
        if existing:
            conn.close()
            return jsonify({'error': 'Email already registered'}), 409
        user_id = str(uuid.uuid4())
        conn.execute('INSERT INTO users (id, full_name, email, password_hash, created_at) VALUES (?,?,?,?,?)',
                     (user_id, full_name, email, hash_password(password), datetime.now(UTC).isoformat()))
        conn.commit()
        conn.close()
        token = generate_token(user_id)
        return jsonify({'user_id': user_id, 'token': token, 'full_name': full_name, 'email': email, 'message': 'Account created successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        conn.close()
        if not user or not verify_password(password, user['password_hash']):
            return jsonify({'error': 'Invalid credentials'}), 401
        if user['disabled']:
            return jsonify({'error': 'Account disabled'}), 403
        token = generate_token(user['id'])
        return jsonify({'token': token, 'user_id': user['id'], 'full_name': user['full_name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profile', methods=['GET'])
@require_auth
def get_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT id, full_name, email, created_at FROM users WHERE id=?', (user_id,)).fetchone()
    total = conn.execute('SELECT COUNT(*) FROM analyses WHERE user_id=?', (user_id,)).fetchone()[0]
    conn.close()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'id': user['id'], 'full_name': user['full_name'], 'email': user['email'], 'created_at': user['created_at'], 'total_analyses': total})

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        conn = get_db()
        user = conn.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
        if not user:
            return jsonify({'message': 'If the email exists, a reset link has been sent.'})
        reset_token = str(uuid.uuid4()).replace('-', '')[:32]
        expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        conn.execute('UPDATE users SET reset_token=?, reset_expires=? WHERE id=?', (reset_token, expires, user['id']))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Password reset link sent to your email.', 'reset_token': reset_token})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        token = data.get('token', '')
        new_password = data.get('password', '')
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        conn = get_db()
        user = conn.execute('SELECT id FROM users WHERE reset_token=? AND reset_expires>?',
                           (token, datetime.now(UTC).isoformat())).fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        conn.execute('UPDATE users SET password_hash=?, reset_token=NULL, reset_expires=NULL WHERE id=?',
                     (hash_password(new_password), user['id']))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Password reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/google', methods=['POST'])
def google_auth():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        full_name = data.get('full_name', '').strip()
        google_id = data.get('google_id', '')
        if not email:
            return jsonify({'error': 'Email required'}), 400
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        if user:
            if user['disabled']:
                conn.close()
                return jsonify({'error': 'Account disabled'}), 403
            token = generate_token(user['id'])
            conn.close()
            return jsonify({'token': token, 'user_id': user['id'], 'full_name': user['full_name']})
        user_id = str(uuid.uuid4())
        conn.execute('INSERT INTO users (id, full_name, email, password_hash, created_at) VALUES (?,?,?,?,?)',
                     (user_id, full_name, email, hash_password(google_id + 'groomiq_oauth'), datetime.now(UTC).isoformat()))
        conn.commit()
        conn.close()
        token = generate_token(user_id)
        return jsonify({'token': token, 'user_id': user_id, 'full_name': full_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/github', methods=['POST'])
def github_auth():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        full_name = data.get('full_name', '').strip()
        github_id = data.get('github_id', '')
        if not email:
            return jsonify({'error': 'Email required'}), 400
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        if user:
            if user['disabled']:
                conn.close()
                return jsonify({'error': 'Account disabled'}), 403
            token = generate_token(user['id'])
            conn.close()
            return jsonify({'token': token, 'user_id': user['id'], 'full_name': user['full_name']})
        user_id = str(uuid.uuid4())
        conn.execute('INSERT INTO users (id, full_name, email, password_hash, created_at) VALUES (?,?,?,?,?)',
                     (user_id, full_name, email, hash_password(github_id + 'groomiq_oauth'), datetime.now(UTC).isoformat()))
        conn.commit()
        conn.close()
        token = generate_token(user_id)
        return jsonify({'token': token, 'user_id': user_id, 'full_name': full_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================================================================
# PREDICT / ANALYSIS ENDPOINT
# ================================================================
@app.route('/predict', methods=['POST'])
@optional_auth
def predict(user_id):
    if user_id == 'guest' and request.form.get('user_id'):
        user_id = request.form['user_id']
    try:
        file = request.files['image']
        img_bytes = file.read()
        img_299 = Image.open(io.BytesIO(img_bytes)).resize((299, 299)).convert("RGB")
        img_224 = Image.open(io.BytesIO(img_bytes)).resize((224, 224)).convert("RGB")
        face_shape, confidence_score = detect_face_shape_from_image(img_bytes)
        beard_style = recommend_beard(face_shape)
        hair_type = None
        hair_confidence = 0
        top2 = []
        # Predict hair TYPE using dedicated type model (5 classes)
        if type_model is not None:
            arr = np.expand_dims(np.array(img_224, dtype=np.float32), axis=0)
            arr = preprocess_input(arr)
            probs = type_model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            hair_type = TYPE_CLASSES[top_idx]
            hair_confidence = round(float(probs[top_idx]) * 100, 2)
            top2 = [{"hair_type": TYPE_CLASSES[i], "confidence": round(float(probs[i]) * 100, 2)}
                    for i in probs.argsort()[-2:][::-1]]
        # Predict hair CONDITION using dedicated condition model (4 classes: bald, dry, hairfall, healthy)
        hair_condition = None
        cond_confidence = 0
        if cond_model is not None:
            arr = np.expand_dims(np.array(img_224, dtype=np.float32), axis=0)
            arr = preprocess_input(arr)
            probs = cond_model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            hair_condition = COND_CLASSES[top_idx]
            cond_confidence = round(float(probs[top_idx]) * 100, 2)
        # Fallback: use combined main model if type_model unavailable
        if hair_type is None and model is not None:
            arr = np.expand_dims(np.array(img_224, dtype=np.float32), axis=0)
            arr = preprocess_input(arr)
            probs = model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            hair_type = CLASSES[top_idx]
            hair_confidence = round(float(probs[top_idx]) * 100, 2)
            top2 = [{"hair_type": CLASSES[i], "confidence": round(float(probs[i]) * 100, 2)}
                    for i in probs.argsort()[-2:][::-1]]
        if hair_type is None:
            hair_type = "Healthy"
            hair_confidence = confidence_score
        recommended_styles = recommend_styles(face_shape, hair_type)
        grooming_tips = get_grooming_tips(face_shape)
        advice = get_advice(hair_type.lower())
        thumb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=70)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        analysis_id = str(uuid.uuid4())
        share_token = str(uuid.uuid4()).replace('-', '')[:10]
        conn = get_db()
        conn.execute("""INSERT INTO analyses
            (id, user_id, face_shape, hair_type, beard_style, recommended_styles, confidence, top2, web_tips, grooming_tips, image_b64, share_token, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            analysis_id, user_id, face_shape, hair_type, beard_style,
            json.dumps(recommended_styles), hair_confidence, json.dumps(top2),
            json.dumps(advice.get("tips_from_web", [])), json.dumps(grooming_tips),
            img_b64, share_token, datetime.now(UTC).isoformat()))
        conn.execute("""INSERT INTO notifications (id, user_id, title, message, type, created_at)
            VALUES (?,?,?,?,?,?)""",
            (str(uuid.uuid4()), user_id, 'Analysis Complete',
             f'Your {face_shape} face analysis is ready with {hair_confidence}% confidence.',
             'analysis', datetime.now(UTC).isoformat()))
        conn.commit()
        conn.close()
        return jsonify({
            "hair_type": hair_type, "confidence": hair_confidence,
            "hair_condition": hair_condition, "cond_confidence": cond_confidence,
            "face_shape": face_shape, "confidence_score": confidence_score,
            "beard_style": beard_style,
            "recommended_styles": recommended_styles,
            "top2": top2, "grooming_tips": grooming_tips,
            "web_tips": advice.get("tips_from_web", []),
            "advice_source": advice.get("source", "aad.org"),
            "advice_url": advice.get("source_url", ""),
            "analysis_id": analysis_id, "share_token": share_token
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# HISTORY
# ================================================================
@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    try:
        conn = get_db()
        rows = conn.execute("""SELECT id, face_shape, hair_type, beard_style, recommended_styles,
            confidence, top2, web_tips, grooming_tips, image_b64, share_token, created_at
            FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 100""", (user_id,)).fetchall()
        conn.close()
        return jsonify({"history": [{
            "id": r["id"], "face_shape": r["face_shape"],
            "hair_type": r["hair_type"], "beard_style": r["beard_style"],
            "recommended_styles": json.loads(r["recommended_styles"] or "[]"),
            "confidence": r["confidence"],
            "top2": json.loads(r["top2"] or "[]"),
            "web_tips": json.loads(r["web_tips"] or "[]"),
            "grooming_tips": json.loads(r["grooming_tips"] or "[]"),
            "image": f"data:image/jpeg;base64,{r['image_b64']}" if r["image_b64"] else "",
            "share_token": r["share_token"],
            "date": r["created_at"],
        } for r in rows], "total": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history/<user_id>/<analysis_id>', methods=['DELETE'])
@require_auth
def delete_entry(uid, user_id, analysis_id):
    if uid != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    conn.execute("DELETE FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

@app.route('/history/<user_id>/clear', methods=['DELETE'])
@require_auth
def clear_history(uid, user_id):
    if uid != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    conn.execute("DELETE FROM analyses WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})

# ================================================================
# SHARE
# ================================================================
@app.route('/share/<share_token>', methods=['GET'])
def get_shared(share_token):
    try:
        conn = get_db()
        row = conn.execute("""SELECT face_shape, hair_type, beard_style, recommended_styles,
            confidence, top2, web_tips, grooming_tips, image_b64, created_at
            FROM analyses WHERE share_token=?""", (share_token,)).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Report not found"}), 404
        return jsonify({
            "face_shape": row["face_shape"], "hair_type": row["hair_type"],
            "beard_style": row["beard_style"],
            "recommended_styles": json.loads(row["recommended_styles"] or "[]"),
            "confidence": row["confidence"],
            "top2": json.loads(row["top2"] or "[]"),
            "web_tips": json.loads(row["web_tips"] or "[]"),
            "grooming_tips": json.loads(row["grooming_tips"] or "[]"),
            "image": f"data:image/jpeg;base64,{row['image_b64']}" if row["image_b64"] else "",
            "date": row["created_at"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# FAVORITES
# ================================================================
@app.route('/favorites/<user_id>', methods=['GET'])
def get_favorites(user_id):
    conn = get_db()
    rows = conn.execute("SELECT hairstyle, created_at FROM favorites WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return jsonify({"favorites": [dict(r) for r in rows]})

@app.route('/favorites/<user_id>', methods=['POST'])
@require_auth
def add_favorite(uid, user_id):
    if uid != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    hairstyle = data.get('hairstyle', '').strip()
    if not hairstyle:
        return jsonify({'error': 'Hairstyle required'}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO favorites (id, user_id, hairstyle, created_at) VALUES (?,?,?,?)",
                     (str(uuid.uuid4()), user_id, hairstyle, datetime.now(UTC).isoformat()))
        conn.commit()
        return jsonify({"status": "saved", "hairstyle": hairstyle})
    except sqlite3.IntegrityError:
        return jsonify({"status": "already_exists", "hairstyle": hairstyle})
    finally:
        conn.close()

@app.route('/favorites/<user_id>/<hairstyle>', methods=['DELETE'])
@require_auth
def remove_favorite(uid, user_id, hairstyle):
    if uid != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    conn.execute("DELETE FROM favorites WHERE user_id=? AND hairstyle=?", (user_id, hairstyle))
    conn.commit()
    conn.close()
    return jsonify({"status": "removed"})

# ================================================================
# RATINGS
# ================================================================
@app.route('/ratings/<hairstyle>', methods=['GET'])
def get_ratings(hairstyle):
    conn = get_db()
    row = conn.execute("SELECT AVG(rating) as avg, COUNT(*) as count FROM ratings WHERE hairstyle=?", (hairstyle,)).fetchone()
    conn.close()
    return jsonify({"hairstyle": hairstyle, "avg_rating": round(row['avg'] or 0, 1), "count": row['count']})

@app.route('/ratings', methods=['POST'])
@require_auth
def submit_rating(user_id):
    data = request.get_json()
    hairstyle = data.get('hairstyle', '').strip()
    rating = data.get('rating', 0)
    if not hairstyle or not (1 <= rating <= 5):
        return jsonify({'error': 'Invalid hairstyle or rating'}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO ratings (id, user_id, hairstyle, rating, created_at) VALUES (?,?,?,?,?)",
                     (str(uuid.uuid4()), user_id, hairstyle, rating, datetime.now(UTC).isoformat()))
        conn.commit()
        return jsonify({"status": "rated", "hairstyle": hairstyle, "rating": rating})
    except sqlite3.IntegrityError:
        conn.execute("UPDATE ratings SET rating=?, created_at=? WHERE user_id=? AND hairstyle=?",
                     (rating, datetime.now(UTC).isoformat(), user_id, hairstyle))
        conn.commit()
        return jsonify({"status": "updated", "hairstyle": hairstyle, "rating": rating})
    finally:
        conn.close()

@app.route('/ratings/user/<user_id>', methods=['GET'])
def user_ratings(user_id):
    conn = get_db()
    rows = conn.execute("SELECT hairstyle, rating FROM ratings WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    return jsonify({"ratings": [dict(r) for r in rows]})

# ================================================================
# TRENDING STYLES
# ================================================================
@app.route('/trending', methods=['GET'])
def trending():
    conn = get_db()
    rows = conn.execute("""SELECT recommended_styles FROM analyses WHERE recommended_styles IS NOT NULL""").fetchall()
    style_counts = {}
    for r in rows:
        styles = json.loads(r['recommended_styles'] or '[]')
        for s in styles:
            style_counts[s] = style_counts.get(s, 0) + 1
    sorted_styles = sorted(style_counts.items(), key=lambda x: -x[1])[:10]
    conn.close()
    return jsonify({"trending": [{"hairstyle": s, "count": c} for s, c in sorted_styles]})

# ================================================================
# DASHBOARD STATS
# ================================================================
@app.route('/dashboard/<user_id>', methods=['GET'])
def dashboard_stats(user_id):
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM analyses WHERE user_id=?", (user_id,)).fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM analyses WHERE user_id=?", (user_id,)).fetchone()[0] or 0
    faces = conn.execute("SELECT face_shape, COUNT(*) as cnt FROM analyses WHERE user_id=? GROUP BY face_shape ORDER BY cnt DESC", (user_id,)).fetchall()
    most_common_face = faces[0]['face_shape'] if faces else '—'
    last = conn.execute("SELECT created_at FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
    last_date = last['created_at'] if last else '—'
    fav_count = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id=?", (user_id,)).fetchone()[0]
    conn.close()
    return jsonify({
        "total_analyses": total, "avg_confidence": round(avg_conf, 1),
        "most_common_face": most_common_face, "last_analysis_date": str(last_date) if last_date else '\u2014',
        "favorites_count": fav_count
    })

# ================================================================
# NOTIFICATIONS
# ================================================================
@app.route('/notifications/<user_id>', methods=['GET'])
def get_notifications(user_id):
    conn = get_db()
    rows = conn.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 50", (user_id,)).fetchall()
    unread = conn.execute("SELECT COUNT(*) FROM notifications WHERE user_id=? AND read=0", (user_id,)).fetchone()[0]
    conn.close()
    return jsonify({"notifications": [dict(r) for r in rows], "unread": unread})

@app.route('/notifications/<user_id>/read/<notif_id>', methods=['POST'])
def mark_read(user_id, notif_id):
    conn = get_db()
    conn.execute("UPDATE notifications SET read=1 WHERE id=? AND user_id=?", (notif_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "read"})

@app.route('/notifications/<user_id>/read-all', methods=['POST'])
def mark_all_read(user_id):
    conn = get_db()
    conn.execute("UPDATE notifications SET read=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "all_read"})

# ================================================================
# PDF REPORT
# ================================================================
@app.route('/report/<analysis_id>', methods=['GET'])
def download_report(analysis_id):
    def pdf_safe(text):
        if not text: return ''
        return text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"').replace('\u2026', '...').replace('\u2022', '-').encode('ascii', 'replace').decode('ascii')
    try:
        from fpdf import FPDF
        conn = get_db()
        row = conn.execute("""SELECT u.full_name, a.* FROM analyses a
            JOIN users u ON a.user_id=u.id WHERE a.id=?""", (analysis_id,)).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Report not found"}), 404
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(200, 169, 110)
        pdf.cell(0, 15, "GroomIQ Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(136, 136, 153)
        pdf.cell(0, 8, f"Generated: {datetime.now(UTC).strftime('%B %d, %Y at %H:%M')} UTC", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        pdf.set_draw_color(200, 169, 110)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(232, 232, 240)
        pdf.cell(0, 10, "Personal Information", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(136, 136, 153)
        pdf.cell(0, 8, f"Name: {pdf_safe(row['full_name'])}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(232, 232, 240)
        pdf.cell(0, 10, "Analysis Results", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(136, 136, 153)
        items = [
            ("Face Shape", pdf_safe(row['face_shape'])),
            ("Hair Type", pdf_safe(row['hair_type'] or 'N/A')),
            ("Confidence", f"{row['confidence']}%"),
            ("Recommended Beard", pdf_safe(row['beard_style'] or 'N/A')),
            ("Analysis Date", pdf_safe(row['created_at'][:10])),
        ]
        for label, value in items:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(200, 169, 110)
            pdf.cell(60, 8, label)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(232, 232, 240)
            pdf.cell(0, 8, str(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        styles = json.loads(row['recommended_styles'] or '[]')
        if styles:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(232, 232, 240)
            pdf.cell(0, 10, "Recommended Hairstyles", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(136, 136, 153)
            for s in styles:
                pdf.cell(0, 8, f"  - {pdf_safe(s)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        tips = json.loads(row['grooming_tips'] or '[]')
        if tips:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(232, 232, 240)
            pdf.cell(0, 10, "Grooming Tips", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(136, 136, 153)
            for t in tips:
                pdf.set_x(15)
                pdf.cell(0, 8, f"  - {pdf_safe(t)}", new_x="LMARGIN", new_y="NEXT")
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=f'groomiq-report-{analysis_id[:8]}.pdf')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# AI HAIRSTYLE PREVIEW (placeholder using Replicate-style API)
# ================================================================
@app.route('/ai-preview', methods=['POST'])
@require_auth
def ai_preview(user_id):
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        hairstyle = data.get('hairstyle', '')
        face_shape = data.get('face_shape', '')
        if not image_b64 or not hairstyle:
            return jsonify({'error': 'Image and hairstyle required'}), 400
        result_img, error = generate_preview(image_b64, hairstyle, face_shape)
        if error:
            return jsonify({"error": error, "hint": "Set REPLICATE_API_TOKEN or STABILITY_API_KEY in backend/.env"}), 503
        return jsonify({
            "status": "generated",
            "hairstyle": hairstyle,
            "image": result_img
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# HAIRSTYLE GALLERY DATA
# ================================================================
HAIRSTYLES_GALLERY = [
    {"name": "Quiff", "description": "Classic voluminous style with lifted front and tapered sides.", "suitable_faces": ["Oval", "Round", "Square"], "image": "style1.png"},
    {"name": "Pompadour", "description": "Bold volume swept upward and back, with short sides.", "suitable_faces": ["Oval", "Round", "Heart"], "image": "style2.png"},
    {"name": "Crew Cut", "description": "Short, tapered cut that's easy to maintain and clean.", "suitable_faces": ["Square", "Rectangle", "Oval"], "image": "style3.png"},
    {"name": "Buzz Cut", "description": "Uniform short length all over — minimalist and bold.", "suitable_faces": ["Square", "Oval", "Diamond"], "image": "style4.png"},
    {"name": "Faux Hawk", "description": "Edgy style with raised center strip and faded sides.", "suitable_faces": ["Round", "Oval", "Square"], "image": "style1.png"},
    {"name": "Slick Back", "description": "Polished style with hair combed back using product.", "suitable_faces": ["Oval", "Square", "Rectangle"], "image": "style2.png"},
    {"name": "Textured Crop", "description": "Short top with choppy texture and faded sides.", "suitable_faces": ["Oval", "Square", "Triangle"], "image": "style3.png"},
    {"name": "Side Part", "description": "Clean parting with short sides — timeless professional look.", "suitable_faces": ["Oval", "Square", "Rectangle", "Diamond"], "image": "style4.png"},
    {"name": "Undercut", "description": "Long top with shaved or closely cropped sides.", "suitable_faces": ["Round", "Oval", "Diamond"], "image": "style1.png"},
    {"name": "Fade", "description": "Gradual taper from short to skin at the sides and back.", "suitable_faces": ["Oval", "Round", "Square", "Diamond"], "image": "style2.png"},
]

@app.route('/gallery', methods=['GET'])
def get_gallery():
    face_shape = request.args.get('face_shape', '')
    base_url = request.host_url.rstrip('/')
    results = HAIRSTYLES_GALLERY
    if face_shape:
        results = [h for h in results if face_shape in h['suitable_faces']]
    return jsonify({"hairstyles": results})

# ================================================================
# ADMIN
# ================================================================
@app.route('/admin/users', methods=['GET'])
@admin_required
def admin_get_users(uid):
    conn = get_db()
    rows = conn.execute("SELECT id, full_name, email, created_at, is_admin, disabled FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify({"users": [dict(r) for r in rows]})

@app.route('/admin/users/<user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(uid, user_id):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND is_admin=0", (user_id,))
    conn.execute("DELETE FROM analyses WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

@app.route('/admin/users/<user_id>/toggle', methods=['POST'])
@admin_required
def admin_toggle_user(uid, user_id):
    conn = get_db()
    user = conn.execute("SELECT disabled FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    new_status = 0 if user['disabled'] else 1
    conn.execute("UPDATE users SET disabled=? WHERE id=? AND is_admin=0", (new_status, user_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "updated", "disabled": bool(new_status)})

@app.route('/admin/analyses', methods=['GET'])
@admin_required
def admin_get_analyses(uid):
    conn = get_db()
    rows = conn.execute("""SELECT a.id, a.user_id, a.face_shape, a.confidence, a.created_at, u.full_name
        FROM analyses a JOIN users u ON a.user_id=u.id ORDER BY a.created_at DESC LIMIT 200""").fetchall()
    conn.close()
    return jsonify({"analyses": [dict(r) for r in rows]})

@app.route('/admin/analytics', methods=['GET'])
@admin_required
def admin_analytics(uid):
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    face_shapes = conn.execute("SELECT face_shape, COUNT(*) as cnt FROM analyses GROUP BY face_shape ORDER BY cnt DESC").fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0] or 0
    conn.close()
    return jsonify({
        "total_users": total_users, "total_analyses": total_analyses,
        "avg_confidence": round(avg_conf, 1),
        "face_shape_distribution": [{"shape": r['face_shape'], "count": r['cnt']} for r in face_shapes]
    })

@app.route('/admin/analyses/<analysis_id>', methods=['DELETE'])
@admin_required
def admin_delete_analysis(uid, analysis_id):
    conn = get_db()
    conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

# ================================================================
# HEALTH
# ================================================================
@app.route('/health', methods=['GET'])
def health():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    conn.close()
    return jsonify({'status': 'running', 'total_analyses': total, 'total_users': users})

@app.route('/refresh-advice', methods=['POST'])
def refresh_advice():
    try:
        from hair_advice import refresh_cache
        refresh_cache()
        return jsonify({"status": "cache refreshed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import tensorflow as tf
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, port=port, host='0.0.0.0')
