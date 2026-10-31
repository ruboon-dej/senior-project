# ThaiSL Translator — Mockup

A Google-Translate-style UI for Thai ↔ Thai Sign Language (ThSL), built as
two independent projects so the frontend and backend can be developed and
deployed separately.

```
thsl-translator/
├── frontend/   Next.js + TypeScript + Tailwind (the UI)
└── backend/    FastAPI (mock translation endpoints, ready to plug in your model)
```

## Why this stack

- **Frontend — Next.js + TypeScript + Tailwind**: works great on desktop
  today, and since it's just React under the hood, making it responsive for
  mobile later is a styling change, not a rewrite.
- **Backend — FastAPI (Python)**: your existing Thai2ThSL pipeline
  (`thai2thsl/app.py` in the model repo) is already Python/TensorFlow, so
  the backend can eventually import that code directly instead of needing a
  translation layer between languages.
- **Separation**: the frontend only knows about two REST endpoints
  (`/api/translate/thai-to-thsl` and `/api/translate/thsl-to-thai`). Nothing
  in the frontend needs to change when the backend swaps from mock data to
  your real model — see the `TODO` comments in
  `backend/routers/translate.py`.

## Running it locally

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs (auto-generated
by FastAPI).

#### Running Backend the first tome
Please install e-tsl-pose.tar, convert them to normal folder, and put it in the backend directory.
Additionally, download the model-datasets.tar, convert them, pull out files inside models folder and combine them with the existing files in there, and put the whole datasets folder in the backend directory.

These 2 .tar files are accesible via https://drive.google.com/drive/folders/1Qu6MWlXaRNMoeyAtvIbUBBYnpAR0o-gS

### Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

#### When running Frontend for the first time
Please check whther nvm is installed yet. via
`command -v nvm`
See if there is anything returned or not.

If nothing is installed, run
`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash`

Then
`source ~/.zshrc`

Visit `http://localhost:3000`.

## What's mocked right now

- `POST /api/translate/thai-to-thsl` — splits the input text into words as
  fake "glosses" and always returns the same sample video URL.
- `POST /api/translate/thsl-to-thai` — reads the uploaded video (to prove
  the upload works) but always returns the same placeholder Thai sentence.
- The microphone button on the Thai→ThSL page and the camera-record button
  on the ThSL→Thai page are visual placeholders (not wired to the Web
  Speech API / MediaRecorder yet).

## What to build next

1. Replace the mock logic in `backend/routers/translate.py` with real calls
   into your Thai2ThSL / ThSL2Thai pipelines.
2. Decide how generated videos are served (e.g. save to disk / object
   storage and return a URL, or stream bytes directly).
3. Wire up the microphone button (Web Speech API) and the record button
   (MediaRecorder API) on the frontend.
4. If you want the "select desired gloss/symbol" disambiguation step from
   your sketch (for ambiguous words), that would slot in as a step between
   submitting text and playing the video — happy to help design that when
   you're ready.
5. Add authentication / rate limiting before deploying publicly.
