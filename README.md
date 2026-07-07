# Smart Baby Monitor — Complete Project Documentation

## 1. Project Overview

### 1.1 Title
**Smart Baby Monitor** — An intelligent real-time nursery monitoring system

### 1.2 Institution & Context
Final-year **Software Engineering** project at **JCE** (Jerusalem College of Engineering). The system is designed to help parents and caregivers supervise an infant remotely using a combination of video, audio, on-device machine learning, and mobile notifications.

### 1.3 Problem Statement
Traditional baby monitors provide audio/video only. Parents must actively watch or listen at all times. This project addresses the gap by:
- Automatically detecting meaningful events (crying, movement, posture changes, sleep/wake transitions)
- Sending configurable alerts to a mobile phone even when the app is not in the foreground
- Tracking sleep patterns over time for basic analytics
- Enabling remote monitoring over the internet without complex network setup

### 1.4 Solution Summary
A **client–server architecture**:
- **Server (Raspberry Pi 5):** Captures camera and microphone input, runs ML models locally, controls a pan/tilt gimbal, stores sleep data, and exposes a REST/HTTP API.
- **Client (Android app):** Displays live video and audio, shows cry/sleep status, controls the camera gimbal, configures alerts, and receives push notifications via a background foreground service.

### 1.5 Key Capabilities (Feature List)

| Feature | Description |
|---------|-------------|
| Live video streaming | MJPEG over HTTP at 640×480, with pose and sleep overlay |
| Live audio streaming | Raw PCM 16 kHz mono over HTTP |
| Pose estimation | Lying / sitting / standing classification from body keypoints |
| Sleep tracking | Debounced states: sleeping, awake, out_of_frame |
| Sleep analytics | 24-hour timeline + daily/weekly/monthly summaries in SQLite |
| Cry detection | YAMNet TFLite audio classifier (baby cry + crying classes) |
| Sound metering | RMS microphone level for generic sound alerts |
| Motion gating | Skips expensive YOLO inference when the scene is static (reduces Pi heat/CPU) |
| Pan/tilt gimbal | Two SG90 servos on hardware PWM; manual joystick + auto-tracking |
| BLE Wi-Fi provisioning | First-time setup: phone connects over Bluetooth, sends Wi-Fi credentials to Pi |
| Remote viewing | Tailscale VPN — parents view nursery from anywhere via private IP |
| Alert system | Configurable push notifications (sound, cry, motion, baby movement, posture) |
| Background monitoring | Android foreground service polls server every 3 seconds when enabled |

---

## 2. System Architecture

### 2.1 High-Level Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5 (Server)                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Camera   │  │ USB Mic  │  │ SG90 Servos│  │  Bluetooth  │  │
│  │ OV5647   │  │ 16 kHz   │  │ Pan / Tilt │  │  (BLE setup)│  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └──────┬──────┘  │
│       │             │              │                 │          │
│  ┌────▼─────────────▼──────────────▼─────────────────▼──────┐  │
│  │                    Flask HTTP Server (:5001)               │  │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │  │
│  │  │ Vision  │ │  Audio   │ │ Hardware │ │ Provisioning  │  │  │
│  │  │ Pipeline│ │ Pipeline │ │  Gimbal  │ │  BLE + Wi-Fi  │  │  │
│  │  └────┬────┘ └────┬─────┘ └──────────┘ └───────────────┘  │  │
│  │       │           │                                          │  │
│  │  YOLOv8 Pose   YAMNet TFLite                                │  │
│  │  Sleep Tracker  Sound Meter                                  │  │
│  │  Inference Gate  Cry Detector                               │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────┘
                              │ HTTP (local Wi-Fi or Tailscale VPN)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Android App (Client)                          │
│  Live video (WebView MJPEG) · Live audio (AudioTrack PCM)       │
│  Gimbal joystick · Cry status · Sleep analytics · Alert options │
│  NurseryAlertService (foreground) → push notifications          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Communication Modes

| Mode | When used | How |
|------|-----------|-----|
| **Local Wi-Fi** | Phone and Pi on same home network | Direct HTTP to Pi LAN IP (e.g. `http://192.168.x.x:5001`) |
| **Remote (Internet)** | Parent away from home | Tailscale mesh VPN; app uses Pi Tailscale IP (`100.x.x.x`) |
| **BLE provisioning** | First-time setup only | Phone sends Wi-Fi SSID/password over Bluetooth LE; Pi connects via NetworkManager |

The Android app labels remote mode **"Internet"** in the user interface (not "Tailscale").

### 2.3 Threading Model (Server)

The server uses multiple concurrent threads:
1. **Flask main thread** — handles HTTP requests (video stream blocks one worker thread per client)
2. **Camera capture loop** — inside `generate_frames()`: captures frames, motion gating, enqueues inference
3. **Inference worker thread** — runs YOLO pose model on downscaled frames (~320×240), updates pose/sleep state
4. **Audio capture hub** — single microphone capture thread fans out PCM to HTTP subscribers, cry detector, and sound meter
5. **BLE provisioning thread** — optional background BlueZ GATT server

---

## 3. Hardware

| Component | Role | Notes |
|-----------|------|-------|
| **Raspberry Pi 5** | Main compute unit | Runs all server software |
| **Raspberry Pi Camera Module (OV5647)** | Video input | 640×480, ~75° horizontal FOV |
| **USB microphone** | Audio input | ALSA device e.g. `plughw:2,0`; 16 kHz mono |
| **2× SG90 servo motors** | Pan/tilt gimbal | Hardware PWM on GPIO 12 (pan) and GPIO 13 (tilt) |
| **Android smartphone** | Client device | Kotlin app, minimum modern Android with BLE |

---

## 4. Repository Structure

The server codebase lives in **`FinalProject/`**. The Android client lives separately in **`BabyMonitorApp/`** (Android Studio / Kotlin).

### 4.1 Server (`FinalProject/`)

```
FinalProject/
├── run.py                      # Recommended entry point: python run.py
├── web_server.py               # Backward-compatible wrapper (same as run.py)
├── requirements.txt            # Core Python dependencies
├── requirements-ble.txt        # BLE provisioning extras (bluezero)
│
├── config/
│   └── device_config.json      # tailscale_ip, audio_device
│
├── data/                       # Runtime data (gitignored)
│   └── sleep_data.db           # SQLite sleep session history
│
├── models/
│   ├── yolov8n-pose.pt         # Ultralytics YOLOv8 Nano Pose weights
│   ├── yamnet.tflite           # Google YAMNet audio classifier (downloaded)
│   └── yamnet_class_map.csv    # AudioSet class labels for YAMNet
│
├── server/                     # Main Python package
│   ├── app.py                  # Flask app, camera pipeline, all HTTP routes
│   ├── paths.py                # Central path constants (config, models, data)
│   │
│   ├── vision/                 # Camera & computer vision
│   │   ├── frame_buffer.py     # Thread-safe latest frame storage
│   │   ├── inference_gate.py   # Motion-based YOLO skip logic
│   │   ├── pose.py             # Keypoint extraction + posture classification
│   │   ├── sleep_tracker.py    # Sleep/wake state machine + SQLite
│   │   └── auto_tracker.py     # Auto-center camera on detected person
│   │
│   ├── audio/                  # Microphone & audio ML
│   │   ├── stream.py           # Shared mic hub, HTTP /audio_feed
│   │   ├── cry_detector.py     # YAMNet TFLite cry classification
│   │   └── sound_meter.py      # RMS loudness for sound alerts
│   │
│   ├── alerts/
│   │   └── snapshot.py         # Combined audio+camera state for mobile alerts
│   │
│   ├── hardware/
│   │   ├── servo_controller.py # Pan/tilt PWM control (/servo API)
│   │   └── servo_test.py       # Manual keyboard servo test utility
│   │
│   └── provisioning/           # First-time Wi-Fi setup over BLE
│       ├── ble_wifi_provision.py
│       ├── ble_pairing_agent.py
│       ├── ble_chunked_transfer.py
│       ├── device_registry.py
│       ├── wifi_manager.py
│       └── constants.py        # BLE GATT UUIDs
│
├── scripts/
│   └── download_yamnet.py      # One-time YAMNet model download
│
├── docs/
│   └── BLE_SETUP.md            # Pi BLE/Wi-Fi provisioning guide
│
└── deploy/
    ├── polkit/                 # NetworkManager permission rules
    └── sudoers/                # Optional passwordless nmcli for Pi user
```

### 4.2 Android Client (`BabyMonitorApp/`)

Key modules (Kotlin):

| File / Area | Purpose |
|-------------|---------|
| `MainActivity.kt` | Main nursery dashboard: video, audio, gimbal, cry status |
| `FullScreenActivity.kt` | Full-screen live video with gimbal overlay |
| `MonitorApiClient.kt` | HTTP client for all Pi REST endpoints |
| `StreamWebViewHelper.kt` | MJPEG video display in WebView |
| `MonitorAudioPlayer.kt` | Raw PCM playback via AudioTrack |
| `GimbalController.kt` / `JoystickView.kt` | Pan/tilt control via `/servo` |
| `CryStatusMonitor.kt` | Polls `/cry/status` every few seconds |
| `SleepAnalyticsActivity.kt` | Fetches `/sleep/analytics`, displays timeline |
| `WifiSetupActivity.kt` | BLE provisioning flow for first-time Wi-Fi |
| `BleProvisioningClient.kt` | BLE GATT client matching Pi provisioning service |
| `AlertOptionsActivity.kt` | User-facing alert configuration page |
| `NurseryAlertService.kt` | Foreground service; polls `/alerts/snapshot` every 3 s |
| `AlertEngine.kt` | Evaluates snapshot against user thresholds/toggles |
| `AlertNotifier.kt` | Android notification channel + push alerts |
| `AlertPreferences.kt` | SharedPreferences for alert settings |

---

## 5. Server Modules — Technical Detail

### 5.1 Flask Application (`server/app.py`)

**Entry:** `python run.py` or `python web_server.py`  
**Port:** 5001  
**Host:** 0.0.0.0 (all interfaces)

**Startup sequence:**
1. Initialize Picamera2 (640×480 XRGB8888)
2. Load YOLOv8n-pose model from `models/yolov8n-pose.pt`
3. Start BLE Wi-Fi provisioning thread (if BlueZ available)
4. Start audio capture hub + YAMNet cry detector
5. Run Flask with `threaded=True`

**Camera pipeline (per frame):**
1. Capture frame from Picamera2
2. Convert XRGB → BGR for OpenCV
3. Compute motion score via inference gate (160×120 grayscale frame diff)
4. If motion exceeds threshold OR watchdog timer expired → enqueue frame for YOLO inference
5. If no YOLO run → still update sleep tracker from cached keypoints every 0.5 s
6. If auto-track enabled → adjust servos toward person centroid every 0.15 s
7. Overlay pose label, sleep state, FPS on frame
8. Encode JPEG → yield MJPEG multipart stream

### 5.2 Pose Estimation (`server/vision/pose.py` + YOLOv8)

**Model:** Ultralytics **YOLOv8n-pose** (nano — optimized for Raspberry Pi)  
**Input:** Frame downscaled to 320×240 for inference; keypoints scaled back to 640×480  
**Output:** 17 COCO body keypoints per detected person `(x, y, confidence)`

**Posture classification (rule-based heuristics, not a separate ML model):**

| Posture | Logic |
|---------|-------|
| **lying** | Torso angle (shoulder–hip vs vertical) > 50° |
| **sitting** | Knee angle (hip–knee–ankle) < 140° |
| **standing** | Otherwise |
| **unknown** | No person detected or insufficient keypoint confidence |

Key COCO indices used: shoulders (5,6), hips (11,12), knees (13,14), ankles (15,16).

### 5.3 Inference Gate (`server/vision/inference_gate.py`)

**Purpose:** Reduce CPU usage and heat on the Pi by skipping YOLO when unnecessary.

**Method:** Compare consecutive frames downscaled to 160×120 grayscale; compute mean absolute pixel difference (`motion_score`).

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MOTION_THRESHOLD` | 4.0 | Run YOLO if frame diff ≥ this |
| `WATCHDOG_NO_PERSON_SEC` | 4 s | Re-run YOLO even if static when no person visible |
| `WATCHDOG_PERSON_STATIC_SEC` | 12 s | Re-run YOLO if person visible but scene static |

The motion score is also exposed via `/alerts/snapshot` for camera motion alerts.

### 5.4 Sleep Tracker (`server/vision/sleep_tracker.py`)

**States:**
- `sleeping` — baby visible, lying down, low sustained movement
- `awake` — sitting/standing OR high activity index
- `out_of_frame` — fewer than 3 confident core keypoints visible

**Debouncing:** A candidate state must persist before switching:
- sleeping: 10 s · awake: 5 s · out_of_frame: 15 s

**Activity index:** Rolling average of per-frame keypoint displacement over 30 seconds. Prevents brief movements (leg kick, roll) from immediately marking the baby awake.

**Persistence:** SQLite database `data/sleep_data.db`, table `sleep_segments` (state, start_ts, end_ts).

**Analytics API (`/sleep/analytics`):** Returns current status, 24-hour timeline segments, and summary (last 24h sleep/awake minutes, weekly/monthly average sleep hours).

### 5.5 Cry Detection (`server/audio/cry_detector.py`)

**Model:** Google **YAMNet** (TFLite), downloaded via `scripts/download_yamnet.py`  
**Runtime:** `ai-edge-litert` interpreter (Python 3.13 compatible)  
**Audio input:** 16 kHz mono PCM, ~0.975 s windows (15,600 samples)

**Classes monitored (AudioSet indices):**
- Index 19: `Crying, sobbing`
- Index 20: `Baby cry, infant cry`

**State machine:**
- Score threshold: 0.30
- Requires 2 consecutive high-score windows to declare `crying: true`
- Clears after 5 s below threshold
- Debounced `crying` boolean + continuous `score` (0.0–1.0)

**API (`/cry/status`):** `{ available, crying, score, label }`

### 5.6 Audio Streaming (`server/audio/stream.py`)

**Architecture:** Single **capture hub** — one microphone thread captures audio; all consumers receive fan-out copies. This prevents multiple processes competing for the USB mic.

**Capture:** PyAudio callback preferred; falls back to `arecord` subprocess.  
**Format:** 16-bit signed PCM, 16 kHz, mono  
**HTTP endpoint:** `/audio_feed` — raw L16 stream  
**Consumers (always fed, even without HTTP subscribers):** cry detector, sound meter

**Device selection priority:**
1. Environment variable `BABYMONITOR_AUDIO_DEVICE`
2. `config/device_config.json` → `audio_device`
3. Auto-detect USB card via `arecord -l`

### 5.7 Sound Meter (`server/audio/sound_meter.py`)

Computes RMS loudness from PCM chunks (exponentially smoothed). Used by `/alerts/snapshot` for generic "any sound detected" alerts. Values normalized 0.0–1.0.

### 5.8 Alert Snapshot (`server/alerts/snapshot.py`)

**Endpoint:** `GET /alerts/snapshot`  
**Purpose:** Single JSON payload for the Android alert service to evaluate all conditions in one poll.

**Response structure:**
```json
{
  "timestamp": 1717000000.0,
  "audio": {
    "available": true,
    "level": 0.05,
    "peak": 0.12,
    "cry_score": 0.0,
    "crying": false,
    "cry_label": "idle"
  },
  "camera": {
    "available": true,
    "motion_score": 2.1,
    "pose": "lying",
    "sleep_state": "sleeping",
    "activity_index": 3.5,
    "person_visible": true
  }
}
```

### 5.9 Gimbal / Servo Control (`server/hardware/servo_controller.py`)

**Motors:** 2× SG90 hobby servos  
**Control:** Raspberry Pi 5 hardware PWM (`rpi_hardware_pwm`)  
**Range:** −90° to +90° pan and tilt  
**API (`/servo`):**
- `GET` — current angles, availability, auto-track status
- `POST { pan, tilt }` — set absolute angles
- `POST { pan_delta, tilt_delta }` — joystick-style relative step
- `POST { auto_track: true/false }` — enable/disable auto-tracking

**Auto-track (`server/vision/auto_tracker.py`):** Computes centroid of confident keypoints, converts pixel offset to angular error using measured 75° HFOV, applies smoothed fractional steps to avoid oscillation.

### 5.10 BLE Wi-Fi Provisioning (`server/provisioning/`)

**BLE device name:** `BabyMonitor`  
**Flow:**
1. Phone connects over BLE GATT
2. Phone writes `SCAN` command → Pi scans Wi-Fi via NetworkManager (`nmcli`)
3. Pi returns network list (chunked if >512 bytes)
4. Phone sends SSID + password
5. Pi connects via `nmcli`, returns new IP address
6. Pi also exposes Tailscale IP via `/remote/info`

**Headless pairing:** Custom BlueZ agent auto-accepts pairing (no keyboard on Pi).

---

## 6. REST API Reference

| Endpoint | Method | Response | Description |
|----------|--------|----------|-------------|
| `/` | GET | HTML | Simple live video preview page |
| `/video_feed` | GET | MJPEG | Live camera stream with overlays |
| `/audio_feed` | GET | PCM L16 | Live microphone audio |
| `/audio/info` | GET | JSON | Sample rate, device name |
| `/health` | GET | JSON | `{ status, ip, port }` — connectivity check |
| `/pose` | GET | JSON | `{ pose, sleep_state }` |
| `/sleep/status` | GET | JSON | Current sleep tracker state |
| `/sleep/analytics` | GET | JSON | Timeline + 24h/7d/30d summaries |
| `/cry/status` | GET | JSON | Cry detection state and score |
| `/alerts/snapshot` | GET | JSON | Combined alert metrics |
| `/servo` | GET/POST | JSON | Gimbal control |
| `/remote/info` | GET | JSON | `{ tailscale_ip }` for remote viewing |

All JSON status endpoints set `Cache-Control: no-store`.

---

## 7. Android Alert System

### 7.1 Background Monitoring
When the user enables **"Enable nursery alerts"** in Alert Options:
- `NurseryAlertService` starts as an Android **foreground service** (persistent notification required by Android)
- Polls `GET /alerts/snapshot` every **3 seconds**
- Uses local Pi URL if reachable, otherwise Tailscale remote URL
- Service survives app being closed; restarts on app resume if monitoring was enabled

### 7.2 Alert Types

| Category | Alert | Threshold |
|----------|-------|-----------|
| **Sound (master toggle)** | Any sound detected | Sensitivity slider 0–100% |
| **Sound** | Cry detected | Sensitivity slider 0–100% |
| **Camera (master toggle)** | Movement in frame | Sensitivity slider 0–100% |
| **Camera** | Baby movement (activity index) | Sensitivity slider 0–100% |
| **Camera** | Baby lying down | On/off (fires on pose transition) |
| **Camera** | Baby sitting | On/off (fires on pose transition) |
| **Camera** | Baby standing | On/off (fires on pose transition) |

Master toggles disable entire sound or camera sections. Each alert type has a **90-second cooldown** to prevent notification spam.

### 7.3 Permissions
- `POST_NOTIFICATIONS` (Android 13+)
- `FOREGROUND_SERVICE` / `FOREGROUND_SERVICE_DATA_SYNC`

---

## 8. Machine Learning Summary

| Task | Model | Framework | Runs on |
|------|-------|-----------|---------|
| Pose estimation | YOLOv8n-pose | Ultralytics (PyTorch) | Pi CPU |
| Posture classification | Rule-based geometry | Custom Python | Pi CPU |
| Cry detection | YAMNet | TFLite (ai-edge-litert) | Pi CPU |
| Sleep state | Heuristic state machine | Custom Python | Pi CPU |
| Motion gating | Frame differencing | OpenCV + NumPy | Pi CPU |

**Design choice:** All inference runs **on-device on the Pi** — no cloud dependency, lower latency, better privacy for nursery video/audio.

**Optimization:** Motion gating reduces YOLO invocations by ~70–90% when the nursery is static (estimated; depends on scene activity).

---

## 9. Technology Stack

### 9.1 Server (Raspberry Pi)
| Layer | Technology |
|-------|------------|
| Language | Python 3.13 |
| Web framework | Flask |
| Camera | Picamera2 |
| Video processing | OpenCV, NumPy |
| Pose ML | Ultralytics YOLOv8 |
| Audio capture | PyAudio (fallback: ALSA arecord) |
| Audio ML | YAMNet TFLite via ai-edge-litert |
| Database | SQLite3 |
| Servo control | rpi_hardware_pwm |
| BLE | bluezero, python3-dbus, BlueZ |
| Wi-Fi management | NetworkManager (nmcli) |
| Remote access | Tailscale VPN |
| Process management | systemd (`webserver.service`) |

### 9.2 Client (Android)
| Layer | Technology |
|-------|------------|
| Language | Kotlin |
| UI | Android Views + Material |
| Video | WebView (MJPEG) |
| Audio | AudioTrack (raw PCM) |
| Networking | HttpURLConnection |
| BLE | Android Bluetooth LE API |
| Notifications | NotificationCompat + foreground service |
| Storage | SharedPreferences |

### 9.3 Python Dependencies (`requirements.txt`)
```
flask
numpy
opencv-python
picamera2
pyaudio
ai-edge-litert
```
*(Ultralytics/YOLO and rpi_hardware_pwm installed separately on Pi as needed.)*

---

## 10. Configuration

**File:** `config/device_config.json`

```json
{
  "tailscale_ip": "100.106.97.1",
  "audio_device": "plughw:2,0"
}
```

| Key | Purpose |
|-----|---------|
| `tailscale_ip` | Pi's Tailscale IPv4 for remote app viewing |
| `audio_device` | ALSA capture device for USB microphone |

---

## 11. Deployment & Operations

### 11.1 First-Time Pi Setup
```bash
cd ~/Desktop/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ble.txt   # optional, for BLE setup
python scripts/download_yamnet.py     # download cry detection model
python run.py
```

### 11.2 Systemd Auto-Start
The server runs automatically on boot via `webserver.service`:
```bash
sudo systemctl start webserver.service
sudo systemctl stop webserver.service
sudo systemctl status webserver.service
```
Entry point: `python web_server.py` (compatible wrapper for `run.py`).

### 11.3 BLE Wi-Fi Permissions
See `docs/BLE_SETUP.md`. Requires either a polkit rule (`deploy/polkit/`) or sudoers entry (`deploy/sudoers/`) for NetworkManager.

---

## 12. Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| MJPEG over WebRTC | Simpler implementation; works over Tailscale without STUN/TURN servers |
| YOLOv8n (nano) | Best speed/accuracy trade-off on Pi 5 |
| Motion-gated inference | Prevents Pi overheating during static scenes |
| Single audio capture hub | Avoids USB mic contention and audio glitches |
| Rule-based posture from keypoints | No extra model to load; interpretable for medical/demo audience |
| Debounced sleep states | Reduces false awake alerts from brief movements |
| Foreground service for alerts | Required by Android for reliable background polling |
| Tailscale for remote access | No port forwarding; encrypted mesh VPN |
| SQLite for sleep data | Lightweight, no separate DB server on Pi |

---

## 13. Data Privacy & Security Considerations

- Video and audio streams stay on the local network or Tailscale encrypted tunnel — not uploaded to a public cloud
- No user accounts or external authentication in current version (trusted home network model)
- BLE provisioning exposes Wi-Fi credentials only during initial setup over encrypted BLE pairing
- Alert snapshots contain metadata only (scores, states) — not raw video/audio in notifications

---

## 14. Demo & References

**Alpha demo video:**  
https://postjceac-my.sharepoint.com/:v:/g/personal/nirbo_post_jce_ac_il/IQDJ-PZpt7qQRZRPQikN1gN5AX5vDyQBsAnC9I0toYYLZOo

**YAMNet model source:**  
https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite

**YOLOv8 pose:** Ultralytics YOLOv8n-pose (auto-downloaded on first run)

---

## 15. Suggested Report Sections (for Gemini)

When writing a formal report from this document, consider structuring as:

1. **Abstract** — one paragraph summary of problem, solution, and results  
2. **Introduction** — motivation, target users (parents/caregivers), project scope  
3. **Background** — existing baby monitors, ML in childcare, related work  
4. **Requirements** — functional (streaming, alerts, sleep tracking) and non-functional (latency, privacy, Pi constraints)  
5. **System Design** — architecture diagram, client–server split, module descriptions (Section 2, 4, 5)  
6. **Machine Learning** — YOLO pose, YAMNet cry, heuristics (Section 8)  
7. **Implementation** — technologies (Section 9), key algorithms (pose, sleep, motion gate)  
8. **Alert System** — user configuration, background service (Section 7)  
9. **Testing & Evaluation** — demo scenarios, performance on Pi 5, alert accuracy observations  
10. **Conclusion & Future Work** — e.g. multi-camera, cloud backup, improved cry model fine-tuning  
11. **Appendix** — API table (Section 6), project file tree (Section 4)

---

*Document version: June 2026 — Smart Baby Monitor final project, JCE.*
