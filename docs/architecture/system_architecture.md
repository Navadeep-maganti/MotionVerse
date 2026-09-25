# MotionVerse — System Architecture & Tech Stack

## 1. Core Principles & Stack
- **Python 3.11+**
- **Computer Vision**: OpenCV (video capture, frame processing, FPS monitoring), MediaPipe (Pose Landmarker for full-body, Hand Landmarker for hand-mode), NumPy (vectorized math, landmark coordinates, normalizations).
- **Input Simulation**: PyAutoGUI (independent keyboard event dispatch).
- **Desktop UI**: PySide6 (Qt for Python).
- **Configuration**: JSON (game profiles, key bindings, threshold settings).
- **Testing & Quality**: pytest (unit, integration, and performance benchmarks).
- **Packaging**: PyInstaller (offline Windows standalone executable).

---

## 2. Core Architectural Flow

$$\text{Input Modality} \longrightarrow \text{Gesture Recognition} \longrightarrow \text{Generic Action} \longrightarrow \text{Game Mapping} \longrightarrow \text{Keyboard Input} \longrightarrow \text{Active Game}$$

```
Full Body (Pose) ──┐
                   ├──> Generic Action Layer ──> Game Mapper ──> PyAutoGUI ──> Game
Hand (Landmarks) ──┘    (JUMP, CROUCH, MOVE_L,
                         MOVE_R, PAUSE, SELECT)
```

---

## 3. Gesture Recognition & Modalities

### Generic Action Set
Both Full-Body and Hand modes strictly output only generic action signals:
- `JUMP`
- `CROUCH`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `PAUSE`
- `SELECT`

The gesture recognition engines contain **zero game-specific logic**.

### Modalities
1. **Full-Body Mode**:
   - MediaPipe Pose landmarks
   - User-specific calibration & height normalization
   - Relative displacement & velocity/trajectory analysis
   - Adaptive jump/crouch thresholds
   - Temporal filtering & cooldown debouncing
2. **Hand Mode**:
   - MediaPipe Hand landmarks
   - Movement direction & relative coordinates
   - Gesture state machine
   - Temporal filtering & cooldown debouncing

---

## 4. Real-Time Pipeline Design
- Bounded / Latest-Frame buffer to eliminate latency and avoid frame queues.
- Asynchronous separation between frame capture, vision inference, and input dispatch.
- High-precision latency tracking with `time.perf_counter()`.
- Target: 30–60 FPS with minimal end-to-end latency.

---

## 5. UI & Live HUD
- Built with **PySide6**.
- **Pages**: Dashboard, Control Mode Selection, Game Selection, Calibration, Gesture Test, Gaming, Analytics, Settings.
- **Live HUD**: FPS, End-to-end latency, Gesture/Action state, Camera status, Active profile & mode.

