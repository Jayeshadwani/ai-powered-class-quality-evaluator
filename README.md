# Class Quality Evaluator — MVP v0.3 (consistency + severity scoring)

A small, auditable prototype for rating an uploaded children's online class recording. It uses **Streamlit + FastAPI + FFmpeg + local faster-whisper + GPT-4o**. This is an **initial engineering draft, not a validated class-grading system**. It has no speaker diarization, mel-spectrogram analysis, fine-tuning, separate classifier, or direct audio understanding by GPT-4o.

## Agreed design

1. Upload a local video through Streamlit; FastAPI receives the file at `POST /predict`.
2. Use **local FFmpeg** to produce one JPEG approximately every **10 seconds** and 16 kHz mono WAV audio.
3. Use **local faster-whisper** to transcribe the *full recording once*, with VAD and timestamped segments. The transcript is cleaned **only for redundant whitespace**; student replies, fillers, questions, and repetitions are retained.
4. Split into **120-second batches**: normally **12 frames** plus overlapping transcript segments. Preserve actual segment start/end times.
5. For each batch, send JPEGs + cleaned transcript to **GPT-4o**. Get one focused visual observation per frame and evidence for the six rubric signals. **Do not rate the whole class per batch.**
6. After *all* batches, GPT-4o judges each rubric criterion across the **whole class**, considering how often it appears, whether it was fulfilled consistently, and the **severity** of observed problems. Returns `consistent` / `partial` / `not_met` / `unverified`, with **separate positive and negative timestamps**.
7. **Python owns all arithmetic**: within each signal, criteria are equally weighted, worth **100% (consistent), 50% (partial), 0% (not met), 0% (unverified)** of their available points. Signal contributions use the unchanged weights 25/20/20/15/10/10. One minor negative observation does **not** erase positive evidence.
8. Each event is assigned a `criterion_id`; class-level timestamps can earn credit only when grounded in the same criterion and same polarity in the batch evidence. Streamlit shows deductions and their supporting positive/negative timestamps.
   JSON intermediate results are optionally saved locally.

```
MP4 upload → [FFmpeg] → frames every 10 s ─┐
                      └→ WAV → faster-whisper ├→ align 120-s batches
                                           └→ GPT-4o batch evidence (30 calls for 60 minutes)
                                              → GPT-4o whole-class consistency/severity review (1 call)
                                              → deterministic Python weighted score + label
                                              → FastAPI response → Streamlit UI
```

### Agreed signal weights

| Signal | Weight | What the MVP can observe |
| --- | ---: | --- |
| `interaction` | 25% | Individual questions, student participation visible in text (without certain identities) |
| `rapport` | 20% | Greetings, introductions, attention to new students |
| `communication` | 20% | Encouraging/respectful *language*; not actual vocal warmth or prosody |
| `structure` | 15% | Opening, progression, revisions, conclusion |
| `homework` | 10% | Homework and follow-up instructions |
| `visual` | 10% | Background, teacher presence, lighting, camera, material visibility |

Weights and provisional bands live in [`config/rubric.json`](config/rubric.json). The initial label bands are **Poor 0–24**, **Average 25–49**, **Good 50–74**, **Excellent 75–100**. They are **not calibrated**; the labels are a draft convention, not something inferred from the provided feedback. Your feedback uses **Weak** for the lowest category while the challenge requests **Poor**.

## Requirements

- macOS or Linux, **Python 3.11** (required project target).
- **FFmpeg and FFprobe** installed and discoverable on `PATH`.
- An OpenAI API key with access to `gpt-4o` and enough paid API quota.
- Internet on first run to fetch local Whisper weights, and for GPT-4o image/text requests.
- Significant processing time and model charges for 30–60-minute recordings: **30 batch calls + 1 final call for a 60-minute video**, excluding retries. There is no quota safeguard yet.
- Permission from the recording owner/organizer to process children's recordings with the chosen API. User confirmation of this permission is a project precondition.

**Important:** `requirements.txt` pins direct Python dependencies plus the important
transitive compatibility pins `av==17.1.0` and `ctranslate2==4.6.2` for macOS ARM64/Python 3.11.
It is still not a fully resolved cross-platform lockfile. It is **not** a fully reproducible lockfile for transitive packages, FFmpeg executable builds, model weights, Python interpreter binaries or OpenAI model behavior. The remote `gpt-4o` alias may change over time. For strict repeatability later, freeze the full environment and model snapshots.

## Setup (Mac + VS Code Terminal)

From this project's root directory:

```bash
# 1) Install Python 3.11 and FFmpeg (Apple Silicon Mac + Homebrew)
brew install python@3.11 ffmpeg

# 2) Create and activate a fresh environment
# If the prompt says (base), run `conda deactivate` first.
# Explicit interpreter path prevents accidentally using Python 3.15.
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version   # must be Python 3.11.x

# 3) Install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# 4) Create local secret/config file
cp .env.example .env
```

Open `.env` and **replace** `OPENAI_API_KEY=replace_with_your_openai_api_key` with your real key. Do not commit `.env`. Environment defaults support CPU inference without a GPU; `WHISPER_MODEL=small` is a speed-oriented starting point and can make transcription errors, especially on children's speech or varied accents. Changing `WHISPER_LANGUAGE` to an empty value enables language detection.

**Two terminals are needed.** Activate the environment in each.

Terminal 1 — backend:

```bash
source .venv/bin/activate
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
source .venv/bin/activate
python -m streamlit run app/ui.py
```

Open **http://localhost:8501**, upload an MP4/MOV/MKV/WEBM, then click **Assess class**. Video preview is skipped for uploads of **64 MiB or larger** to avoid Tornado static-file cache-version warnings; this does not restrict `/predict` processing. The web app calls **http://127.0.0.1:8000/predict**. Try a short permitted recording first. First transcription can download model weights and take longer. Streamlit's upload limit is set to 1024 MB in `.streamlit/config.toml`; FastAPI separately enforces `MAX_UPLOAD_MB` from `.env`.

### Optional direct API test

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -F 'file=@/absolute/path/to/permitted_class.mp4'
```

Visit `http://127.0.0.1:8000/docs` for the automatically generated API documentation. `/health` checks HTTP process availability (not API key validity or loaded ASR model).

## Troubleshooting: faster-whisper / PyAV `metadata_errors`

`TypeError: open() got an unexpected keyword argument 'metadata_errors'` occurs when a newer PyAV is installed alongside a faster-whisper release that still calls `av.open(..., metadata_errors="ignore")`. PyAV 19 removed that argument. The project pins **`av==17.1.0`** for compatibility with **`faster-whisper==1.1.1`**.

Stop both servers, activate the project's Python 3.11 virtual environment, and reinstall pinned packages:

```bash
conda deactivate  # only if Anaconda (base) is active
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
python -m pip show av faster-whisper
python -m pip check
```

Run `python -m uvicorn app.api:app --host 127.0.0.1 --port 8000` and `python -m streamlit run app/ui.py` from separate terminals, each with `.venv` active. Running `uvicorn` directly from Anaconda base can silently use the wrong interpreter. The `/health` endpoint responds even when transcription dependencies are incompatible; it does not validate ASR.

## Main project structure

```
app/
  api.py              # HTTP upload, size validation, temporary workspace
  ui.py               # Streamlit uploader and output display
  settings.py         # Loads .env, not credentials committed in source
  models.py           # Pydantic request, evidence and result schemas
  ports.py            # Interchangeable service protocols (OCP / DIP)
  factory.py          # Binds concrete implementations in one place
  media.py            # FFprobe + FFmpeg media handling
  transcription.py    # Local faster-whisper adapter + minimal text clean
  batching.py         # 2-minute windows and timestamp alignment
  prompts.py          # Focused batch evidence and final class instructions
  gpt_evaluator.py    # GPT-4o vision/text structured outputs
  rubric.py           # Deterministic weighted score and label bands
  artifacts.py        # Optional local JSON artifacts
  pipeline.py         # Single orchestration path
config/rubric.json    # Editable weights and provisional thresholds
.env.example          # Environment template (safe to commit)
.streamlit/config.toml
requirements.txt
runs/                 # Not committed: sensitive transcript/result JSON
Tests are located in tests/.
```

### Output files

When `SAVE_RESULTS=true`, each successful recording creates:

```text
runs/<random_run_id>/
  transcript.json       # Full timestamped ASR, raw and whitespace-cleaned text
  batch_evidence.json   # Observations for each sampled frame + batch evidence
  assessment.json       # Score / 100, label, per-signal scores and citations to times
```

The uploaded video, extracted audio and JPEG frames are held in a temporary folder that is removed when the request finishes (including on failure). **Transcripts and JSON assessments still contain sensitive content**; protect the local `runs/` folder and delete it when not needed. Setting `SAVE_RESULTS=false` skips persistence of these JSON artifacts. The pipeline does not send raw WAV or MP4 to GPT-4o; it sends only sampled frames and transcript text.

### End-to-end API response (illustrative, shortened)

```json
{
  "run_id": "a1b2...",
  "model": "gpt-4o",
  "duration_seconds": 3600,
  "batch_count": 30,
  "score": 68,
  "label": "Good",
  "weights_version": "mvp-3-consistency-severity",
  "ratings": [
    {
      "signal": "interaction",
      "score": 70,
      "weight": 25,
      "contribution": 17.5,
      "confidence": "medium",
      "rationale": "Some strong and some missed interaction opportunities.",
      "supporting_timestamps": [25.0, 145.0],
      "limitations": "Speaker identities are uncertain.",
      "criterion_deductions": [
        {
          "criterion_id": "individual_questions",
          "status": "partial",
          "issue_severity": "moderate",
          "points_possible": 20.0,
          "points_awarded": 10.0,
          "points_deducted": 10.0,
          "evidence_timestamps": [25.0, 145.0],
          "positive_timestamps": [25.0],
          "negative_timestamps": [145.0],
          "observation": "Questions occurred, but some opportunities were missed."
        }
      ]
    }
  ],
  "overall_observation": "...",
  "recommendations": ["..."]
}
```

The sample is **abbreviated and internally illustrative**. Actual responses contain all six ratings and every configured criterion; the displayed score is not derived from the shortened example.

## Tests

```bash
python -m pytest -q
```

The tests use fake evaluators/transcribers, so they make **no OpenAI API calls**, do not download model weights and do not process real children’s recordings. One test checks FFmpeg with a synthetically generated local test video (skipped when FFmpeg is unavailable). A successful test suite is not equivalent to validating a full paid, uploaded-video flow.

## Explicitly deferred decisions / MVP limitations

1. **Audio delivery:** No raw audio is sent to GPT-4o. It cannot hear tone, pitch, actual vocal enthusiasm or interruptions. Only words and sampled visuals are judged. `communication` therefore measures **verbal expression only**.
2. **No diarization:** It cannot confidently determine which voice is the teacher, student or family member. ASR text alone can show apparent interaction, but do not treat speaker/participant coverage as verified.
3. **Frame sampling:** One frame every 10 seconds cannot prove a camera was continuously on/off or capture all brief events. FFmpeg frame timestamps are **nominal grid times**, which can differ slightly from source presentation timestamps.
4. **Rubric design:** Weights remain exactly as requested. Criterion fulfilment is configured as 100/50/0/0. `not_met` requires grounded negative evidence; `unverified` lacks enough evidence and also earns zero points under the requested evidence-first policy, but is **not** a factual assertion of poor performance. The model judges frequency/severity qualitatively; score bands remain provisional and uncalibrated.
5. **Performance:** `/predict` is **synchronous**, so processing can exceed hosting/gateway timeouts; a background job queue and polling API are deferred. Two-minute windows are processed sequentially for deterministic ordering and easier debugging.
6. **Deploying:** No hosting provider was chosen or deployed. The included commands run both apps on the same development machine. Do not expose this unprotected endpoint publicly with identifiable children's content; authentication, access controls, retention and hosting permission need a deployment plan.
7. **Reliability:** GPT output is schema-validated, with criterion-matched evidence timestamps,
   but factual truth of each observation is not independently verified. We do not yet have held-out accuracy measurements. The Python dependencies are pinned, but ASR model downloads and remote GPT-4o may drift.
8. **VAD:** Default VAD may exclude quiet speech. Sample against actual class audio and retune as needed.
9. **Evaluation cost:** There is no budget limiter or resume-from-failed-batch flow in this initial draft.

## Rationale for code organization

The modules follow **single responsibility**: media decoding, ASR, window construction, LLM inference, scoring, storage and API/UI are independent. `ClassQualityPipeline` depends on typed **Protocols** rather than OpenAI or FFmpeg directly; to use another transcription engine or model implementation, introduce a new class implementing that Protocol and change only the **composition root** (`factory.py`) — a practical application of **open/closed** and dependency inversion.

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Structured Outputs and Pydantic parsing](https://github.com/openai/openai-python/blob/main/helpers.md)
- [FFmpeg](https://ffmpeg.org/ffmpeg.html)
- [Streamlit uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)

## Consistency + severity scoring (v0.3)

**Two-minute GPT-4o calls extract evidence only**, never quality scores. Each event includes a `criterion_id`, timestamp, polarity (`positive` / `negative` / `uncertain`), source and observation. The batch prompt specifically requests both positive behaviours and observed failures; mere silence on a criterion within a batch is **not** a negative event. Evidence from all batches is combined into one ledger, with class-wide counts and batch coverage supplied to the class-level call.

**One class-level GPT-4o call** judges every rubric criterion using all of the evidence. It returns:

| Status | Meaning | Percentage of criterion points |
| --- | --- | ---: |
| `consistent` | Repeated/appropriately timed success; no material recurring weakness | 100% |
| `partial` | Confirmed success but inconsistent or affected by meaningful issues | 50% |
| `not_met` | Grounded significant or repeated shortcomings | 0% |
| `unverified` | Insufficient evidence to decide | 0% |

`issue_severity` is `none`, `minor`, `moderate`, `severe` or `unknown`. This field explains the judgement; **Python does not add an arbitrary second severity penalty**. A single negative event does not automatically veto an entire criterion. Example: five interaction criteria have 20 points each; 3 consistent, 1 partial and 1 unverified yields 70/100 interaction, or 17.5 points of its 25% class weight.

GPT-4o may cite both positive and negative timestamps for the **same** criterion. `GPT4oEvaluator` cross-checks each citation against the ledger with a half-second matching tolerance. Unsupported citations are discarded and any judgement that would earn credit but lacks verified positive citations becomes `unverified`. Direct negative judgements without matching negative evidence are also changed to `unverified`; these outcomes still earn zero. It never guesses evidence to justify a status.

`Rubric` calculates every per-criterion earned point, deduction, dimension score and final score deterministically from [`config/rubric.json`](config/rubric.json). The current 0–24/25–49/50–74/75–100 bands have **not** been calibrated to the supplied human assessments. The aim is to correct the formerly binary scoring contract, not to guarantee that any particular recording will now match its human label.

## Release notes and migration

This complete v0.3 ZIP includes the original FFmpeg/faster-whisper/Streamlit/FastAPI pipeline plus the model/schema, prompts, deterministic rubric, grounding logic, UI, README and test changes for partial credit. **The criteria and six signal weights are unchanged.** The rubric version is `mvp-3-consistency-severity`; the API now reports `positive_timestamps`, `negative_timestamps` and `issue_severity` for each criterion, and status values are different from v0.2. Previously generated assessments should not be compared as if they used identical scoring policies.

Migrate by extracting this ZIP to a **new** project directory and copying only your private `.env` from the old project. Do not copy old `app/`, `config/` or `tests/` folders on top of this version. You can reuse a Python 3.11 virtual environment if it has the pinned dependencies, but a fresh one reduces environment ambiguity. The test suite covers deterministic partial credit, evidence-polarity matching, preservation of negative and positive evidence, API and preprocessing. Tests do not establish end-to-end model accuracy; a real GPT-4o + Whisper run was not performed in the packaging environment.
