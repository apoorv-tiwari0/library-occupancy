# IIT Delhi Library Occupancy Detection & Real-Time Dashboard

A production-grade, real-time computer vision system for monitoring library seating occupancy across 11 designated library sections at IIT Delhi. Uses a zone-based fusion pipeline combining near-zone SAHI+YOLOv10 detection, mid-zone ByteTrack multi-object tracking, and far-zone CSRNet density estimation. Served via a FastAPI backend and a custom React 18 dashboard.

---

## Key Features

- **Zone-Based Detection Pipeline (ZCP-14)**:
  - **Near Zone**: SAHI sliced YOLOv10 object detection for close-up seating.
  - **Mid Zone**: SAHI + YOLOv10 with ByteTrack for tracking and dwell time estimation.
  - **Far Zone**: CSRNet crowd density estimation for distant seating areas.
- **Real-Time Data Streaming**: FastAPI backend supporting REST endpoints and WebSocket live streaming (`/ws/live`).
- **Resilient Dual Storage**: Redis for sub-5ms live state caching and TimescaleDB/PostgreSQL for time-series logging.
- **Edward Tufte-Inspired Dashboard**: Clean, high-density React 18 UI built with Vite, featuring custom gauge progress bars, strict color thresholds, count-up animations, and WebSocket reconnection management.

---

## System Architecture

```
Camera Streams / Video Feeds
           │
           ▼
   Preprocessor & SAHI
           │
 ┌─────────┴─────────┐
 │ Zone Pipeline     │
 │  - Near (YOLO)    │
 │  - Mid (ByteTrack)│
 │  - Far (CSRNet)   │
 └─────────┬─────────┘
           │
           ▼
    Storage Manager  ────────►  Redis (Live Cache: Port 6379)
           │         ────────►  TimescaleDB (History: Port 5433)
           ▼
    FastAPI Backend  (Port 8000)
           │
           ▼
 React Vite Dashboard (Port 5173)
```

---



## Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher with `npm`
- **Docker**: Docker Engine and Docker Compose
- **CUDA** (Optional): NVIDIA GPU with CUDA for accelerated YOLO and CSRNet inference (CPU fallback supported)

---

## Quick Start (Automated One-Command Pipeline)

To run system checks, launch Redis and TimescaleDB storage containers, execute mandatory pre-flight tests, and start both the FastAPI backend and React dashboard in one step:

### Windows (Batch / PowerShell)
```cmd
run_pipeline.bat
```
or
```powershell
.\run_pipeline.ps1
```

### Linux / macOS / Cross-Platform
```bash
python setup_and_deploy.py
```

The script will automatically:
1. Verify system dependencies and config files.
2. Launch Docker storage containers (`docker compose -f docker-compose.db.yml up -d`).
3. Run storage and zone pipeline tests (`tests/test_storage.py`, `tests/test_zcp13.py`).
4. Start the FastAPI backend on `http://localhost:8000`.
5. Start the React dashboard on `http://localhost:5173`.
6. Open your default web browser to the live dashboard.

---

## Manual Installation & Setup

### 1. Environment Setup
Clone the repository and copy the environment template:
```bash
cp .env.example .env
```

Create and activate a Python virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Install frontend dependencies:
```bash
cd library-dashboard
npm install
cd ..
```

### 2. Infrastructure Containers
Start Redis and TimescaleDB containers via Docker Compose:
```bash
docker compose -f docker-compose.db.yml up -d
```

### 3. Run Pre-Flight Tests
```bash
python tests/test_storage.py
python tests/test_zcp13.py
```

### 4. Start the Backend API Server
Launch the FastAPI server with zone-based detection enabled:
```bash
python server.py --zone
```
- API Base URL: `http://localhost:8000`
- Interactive API Documentation: `http://localhost:8000/docs`

### 5. Start the Frontend Dashboard
In a separate terminal window:
```bash
cd library-dashboard
npm run dev
```
- Dashboard UI: `http://localhost:5173`

---

## API Reference

### REST Endpoints

- `GET /health`
  Returns system status, active detection mode (`zone_based` or `standard`), and database connection status.
- `GET /sections`
  Returns current occupancy metrics for all 11 library sections.
- `GET /sections/{section_id}`
  Returns occupancy metrics for a single section.
- `GET /sections/{section_id}/history?hours=24`
  Returns historical time-series data points for the specified section.

### WebSocket Endpoint

- `WS /ws/live`
  Establishes a real-time WebSocket connection. Emits an initial state payload for all sections upon connection, followed by live `occupancy_update` events whenever section states change.

---

## Configuration & Environment Variables

Key parameters configurable in `.env` or `config/config.yaml`:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DB_HOST` | `127.0.0.1` | PostgreSQL / TimescaleDB host |
| `DB_PORT` | `5433` | PostgreSQL port |
| `DB_NAME` | `library_occupancy` | Database name |
| `DB_USER` | `library_user` | Database user |
| `DB_PASSWORD` | `your_db_password` | Database password |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `API_HOST` | `0.0.0.0` | FastAPI server bind host |
| `API_PORT` | `8000` | FastAPI server bind port |
| `USE_ZONE_PIPELINE` | `true` | Enables zone-based fusion detection |
| `MODEL_WEIGHTS_PATH` | `models/yolov10x.pt` | Path to YOLO model weights |
| `MODEL_DEVICE` | `cuda` | PyTorch compute device (`cuda` or `cpu`) |

---

## Project File Structure

```
.
├── config/
│   ├── config.yaml           # Central system configuration
│   └── config_loader.py      # Config parser with env substitution
├── data/
│   └── roi/
│       └── zone_config.json  # Polygon coordinate definitions for zones
├── detection/
│   ├── pipeline.py           # Single and Zone-based detection pipelines
│   ├── zone_detector.py      # Near, Mid, and Far zone detector classes
│   ├── yolo_inference.py     # SAHI + YOLOv10 inference engine
│   ├── density_estimator.py  # CSRNet density estimation wrapper
│   └── csrnet_estimator.py   # CSRNet PyTorch model execution
├── storage/
│   ├── db_logger.py          # TimescaleDB hypertable logger
│   ├── redis_store.py        # Redis live state store
│   └── storage_manager.py    # Unified storage interface
├── library-dashboard/        # React 18 + Vite frontend
│   └── src/
│       ├── api/              # Axios HTTP & WebSocket manager
│       ├── components/       # Header, SummaryBar, SectionCard, etc.
│       ├── hooks/            # Occupancy and timer hooks
│       └── styles/           # CSS modules and design tokens
├── server.py                 # FastAPI backend server
├── setup_and_deploy.py       # Master deployment & testing orchestrator
├── run_pipeline.bat          # Windows launcher script
├── run_pipeline.ps1          # PowerShell launcher script
└── docker-compose.db.yml     # Docker Compose for Redis & TimescaleDB
```

---

## License

Internal research and operational codebase for IIT Delhi Library Occupancy Monitoring.
