# AI Video Ad Generator – Quick‑Deploy Guide

> **Built for:** Chima Full‑Stack Case Study 2025
> **Author:** Anuraag Deoda
> **Stack:** Next.js (React 18 + React‑Bootstrap) · Flask · Python 3.11 · FFmpeg 6 · OpenAI GPT‑4o

---

## ✨ What it does

Scrape any e‑commerce product URL → craft an AI‑generated ad script → render a 15‑30 s MP4 with animated text, product image zoom, and CTA—all from a single page.

---

## 🚀 Get Running in 60 seconds (Docker‑Compose)

```bash
# 1. clone
 git clone https://github.com/yourname/ai‑video‑ad‑generator.git
 cd ai‑video‑ad‑generator

# 2. copy env template & add keys
 cp .env.sample .env               # add OPENAI_API_KEY

# 3. launch everything
 docker compose up --build         # http://localhost:3000
```

*Front‑end on `3000`, Back‑end on `5000`.*

---

## 🛠️ Manual Setup (Local Dev)

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev         # http://localhost:3000
```

### Backend (Flask)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # flask, openai, opencv‑python, ffmpeg‑python
python app.py                     # http://localhost:5000
```

**FFmpeg 6** must be on your PATH (`ffmpeg -version`).

---

## 🔐 Environment Variables (`.env`)

| Key              | Example | Purpose                 |
| ---------------- | ------- | ----------------------- |
| `OPENAI_API_KEY` | sk‑...  | Generate ad scripts     |
| `FLASK_PORT`     | 5000    | Optional override       |
| `CACHE_DURATION` | 300     | Scrape cache in seconds |

---

## 📁 Project Structure (top‑level)

```
.
├─ frontend/          # Next.js 13   (React‑Bootstrap UI)
│  ├─ pages/
│  │   └─ index.tsx   # Orchestrates analyze → generate flow
│  └─ components/     # URLForm, PreviewCard, VideoPlayer, Spinner
│
├─ backend/
│  ├─ app.py          # Flask routes /analyze‑url /generate‑content /generate‑video
│  ├─ scraper.py      # Amazon / Shopify / generic scraping helpers
│  ├─ ai_service.py   # OpenAI wrapper (retry, JSON guarantee)
│  └─ utils.py        # FFmpeg frame builder + helpers
│
├─ data/jobs/         # Saved JSON for each ad job
└─ static/videos/     # Rendered MP4s (served by Flask)
```

*Each backend file is \~200 LoC and documented inline.*

---

## 🖥️ Usage Flow

1. **Paste URL** → frontend POSTs `/api/analyze-url`.
2. **Backend scrape** → returns title, image, price.
3. **Select aspect ratio** (9:16 / 16:9) & hit **Generate Ad**.
4. **OpenAI** crafts script & design cues.
5. **FFmpeg engine** builds frames → encodes MP4.
6. Poll `/api/job-preview/<id>` until ready → stream video.

---

## 🐞 Troubleshooting

| Symptom          | Fix                                                                  |
| ---------------- | -------------------------------------------------------------------- |
| FFmpeg not found | `brew install ffmpeg` · `choco install ffmpeg`                       |
| CORS error       | Ensure frontend `.env` `NEXT_PUBLIC_API_BASE` matches backend port   |
| Blank video      | Check logs for missing image URL; fallback image added in `utils.py` |

---

## 🔭 Roadmap

* [ ] Replace OpenCV/FFmpeg renderer with Remotion for richer animations
* [ ] Add voice‑over generation (AWS Polly)
* [ ] Batch endpoint for multivariate ads

---

> **License:** MIT — feel free to fork, remix, and ship!
