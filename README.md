# GroomIQ — AI Hairstyle Analyzer

Upload a selfie and get personalized hairstyle recommendations, face shape analysis, grooming tips, and AI-powered previews.

## Features

- **Face Shape Detection** — OpenCV Haar cascade + facial proportion analysis (7 shapes with confidence)
- **Hair Type Classification** — EfficientNetB0 model (Straight, Wavy, Curly, Dreadlocks, Kinky) — 93.16% accuracy
- **Hair Condition Analysis** — Detects bald, dry, hairfall, healthy
- **Combined Model** — 11-class fallback when type/condition models unavailable — 77.16% accuracy
- **Hairstyle Recommendations** — Per face-shape & hair-type matched styles
- **Grooming Tips Engine** — Personalized tips for each face shape
- **Beard Style Recommendations** — Matched to face shape
- **AI Preview** — Replicate API (Stability AI fallback) generates hairstyle preview images
- **PDF Reports** — Export analysis as downloadable PDF
- **Google OAuth** — Sign in with Google
- **Email/Password Auth** — JWT-based with forgot password
- **Favorites & Ratings** — Save and rate analyses

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (static) |
| Backend | Python 3.12, Flask, Gunicorn |
| Database | SQLite (dev), PostgreSQL-ready |
| ML Models | TensorFlow / EfficientNetB0 |
| Auth | JWT, bcrypt, Google OAuth |
| AI Preview | Replicate API / Stability AI |
| Deployment | Vercel (frontend), Render (backend) |

## Project Structure

```
├── backend/
│   ├── app.py                 # Flask API (~950 lines)
│   ├── auth/security.py       # JWT, bcrypt, password hashing
│   ├── analysis/
│   │   ├── face_shape.py      # Face shape detection
│   │   └── beard.py           # Beard recommendations
│   ├── recommendations/
│   │   └── hairstyles.py      # Hairstyle matching engine
│   ├── hair_advice.py         # Grooming tips per hair type
│   ├── ai_preview.py          # AI hairstyle preview
│   ├── train_type_model.py    # Type model training script
│   ├── train_condition_model.py
│   ├── train_main_model.py    # Combined model training
│   ├── requirements.txt
│   └── Procfile               # Render entry point
├── frontend/
│   ├── index.html             # Landing page
│   ├── login.html / signup.html
│   ├── dashboard.html         # Main dashboard
│   ├── results.html           # Analysis results
│   ├── history.html           # Past analyses
│   ├── gallery.html           # Hairstyle gallery
│   ├── admin.html             # Admin panel
│   ├── hairstyles/            # Gallery images
│   └── js/
│       ├── config.js          # API URL config
│       └── auth.js            # Auth helpers
├── model/
│   ├── best_model.h5          # Combined model (gitignored)
│   ├── type_model.h5          # Type model (gitignored)
│   └── condition_model.h5     # Condition model (gitignored)
├── datasets/                  # Training data (gitignored)
├── datasets_type/             # Type-specific data (gitignored)
├── datasets_condition/        # Condition-specific data (gitignored)
├── vercel.json                # Vercel deployment config
├── render.yaml                # Render deployment config
└── .env                       # Local env vars (gitignored)
```

## Local Development

### Prerequisites

- Python 3.12+
- TensorFlow-compatible system (Metal on macOS, CUDA on Linux)

### Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/pradnish18/GroomIQ.git
cd GroomIQ

# 2. Create virtual environment
python3 -m venv backend/venv
source backend/venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Create .env file
echo "JWT_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env

# 5. Start backend (port 5001)
backend/venv/bin/python backend/app.py &

# 6. Start frontend (port 8000)
python3 -m http.server 8000 --directory frontend &
```

Open http://localhost:8000 in your browser.

### Training Models

Place dataset images in `datasets/` organized by class folders, then:

```bash
# Type model (5 classes)
backend/venv/bin/python backend/train_type_model.py

# Combined model (11 classes)
backend/venv/bin/python backend/train_main_model.py

# Condition model (4 classes)
backend/venv/bin/python backend/train_condition_model.py
```

Training uses EfficientNetB0 with 224x224 input, 70/20/10 split, batch size 32, Phase 1 (30 epochs) + Phase 2 fine-tuning (20 epochs).

## Deployment

### Frontend — Vercel

1. Push to GitHub
2. Go to https://vercel.com → Import repo → Deploy
3. Vercel auto-detects `vercel.json` (static site, output `frontend/`)
4. Add your Vercel URL to Google Cloud Console OAuth Client ID

### Backend — Render

1. Push to GitHub
2. Go to https://dashboard.render.com → New Web Service → Connect repo
3. `render.yaml` is auto-detected, or use:
   - **Build**: `pip install -r backend/requirements.txt`
   - **Start**: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
4. Set environment variables:
   - `JWT_SECRET_KEY` — run `python3 -c "import secrets; print(secrets.token_hex(32))"`
   - `FLASK_ENV=production`
   - `PYTHON_VERSION=3.12.8`
   - (Optional) `REPLICATE_API_TOKEN` / `STABILITY_API_KEY` for AI preview
   - (Optional) `DATABASE_URL` for PostgreSQL
5. Models are gitignored — upload them separately to Render or use cloud storage

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/signup` | Register with email/password |
| POST | `/login` | Login with email/password |
| POST | `/auth/google` | Google OAuth sign in |
| POST | `/auth/github` | GitHub OAuth (stub) |
| GET | `/profile` | Get user profile |
| POST | `/forgot-password` | Request password reset |
| POST | `/reset-password` | Reset password |
| POST | `/predict` | Upload photo for analysis |
| GET | `/history/<user_id>` | Get analysis history |
| DELETE | `/history/<user_id>/<id>` | Delete analysis |
| POST | `/favorite` | Save/unsave analysis |
| GET | `/favorites/<user_id>` | Get favorites |
| POST | `/rate` | Rate an analysis |
| GET | `/download-pdf/<id>` | Download PDF report |
| POST | `/preview-ai` | Generate AI hairstyle preview |
| GET | `/admin/analyses` | Admin: list all analyses |
| DELETE | `/admin/analyses/<id>` | Admin: delete analysis |
| GET | `/admin/users` | Admin: list users |
| POST | `/admin/users/<id>/toggle` | Admin: disable/enable user |

## Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. APIs & Services → Credentials → Create OAuth 2.0 Client ID
3. Set **Application type**: Web application
4. **Authorized JavaScript origins**: Add your frontend URLs (e.g. `http://localhost:8000`, `https://groom-iq-xi.vercel.app`)
5. **Authorized redirect URIs**: Leave empty (GSI flow, not redirect)
6. Copy the Client ID into `frontend/login.html` and `frontend/signup.html`

## License

MIT
