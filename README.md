# MotionVerse — AI Body Control System

**MotionVerse** is a real-time computer vision application that transforms your body movements into seamless game inputs. Built with **OpenCV**, **MediaPipe**, and **PySide6**, it eliminates the need for physical controllers by using your full body or hands to control character movement, jumping, and crouching.

## 🚀 Key Features

- **Dual Control Modes**:
  - **Full-Body Mode**: Detects full-body posture and large movements for immersive control.
  - **Hand Mode**: Uses hand landmarks for precise, subtle interactions.

- **Advanced Gesture Recognition**:
  - **Jump Detection**: Analyzes both **upward displacement** and **upward velocity** for accurate jumping.
  - **Crouch Detection**: Triggers when **body height compresses** or hips drop downward.
  - **Horizontal Movement**: Smooth left/right steering based on side-to-side motion.

- **Real-Time Performance**:
  - **Latency Optimized**: Strict separation between frame capture, inference, and dispatch.
  - **High FPS**: Target of **30–60 FPS** with minimal end-to-end latency.

- **Full-Featured GUI**:
  - **Dashboard**: Overview of system status and performance metrics.
  - **Control Mode Selection**: Easy switching between Full-Body and Hand modes.
  - **Live HUD**: Real-time display of FPS, latency, gesture states, and camera status.
  - **Calibration**: Interactive calibration for user-specific height and position.
  - **Gesture Testing**: Dedicated window to test and visualize gesture recognition.

- **Configurable & Flexible**:
  - **Game Profiles**: Save and load custom settings for different games.
  - **Key Bindings**: Fully customizable keyboard mapping for all generic actions.
  - **Cross-Platform**: Designed for Windows, macOS, and Linux.

## 🛠️ Technology Stack

- **Core Language**: Python 3.11+
- **Computer Vision**: OpenCV, MediaPipe (Pose & Hands)
- **Input Simulation**: PyAutoGUI
- **Desktop UI**: PySide6 (Qt for Python)
- **Scientific Computing**: NumPy
- **Testing**: pytest
- **Packaging**: PyInstaller

## 📐 Architectural Overview

```mermaid
flowchart LR
    A[Input Modality] --> B[Gesture Recognition]
    B --> C[Generic Action]
    C --> D[Game Mapping]
    D --> E[Keyboard Input]
    E --> F[Active Game]

    subgraph Input Modality
        G[Full Body (Pose)]
        H[Hand (Landmarks)]
    end

    subgraph Generic Action
        I[JUMP]
        J[CROUCH]
        K[MOVE_LEFT]
        L[MOVE_RIGHT]
        M[PAUSE]
        N[SELECT]
    end

    G --> I
    G --> J
    G --> K
    G --> L
    H --> I
    H --> J
    H --> K
    H --> L
    H --> M
    H --> N
```

## 🏃 Controls & Mapping

The system detects generic actions that are then mapped to specific game inputs:

### Generic Actions
- `JUMP`
- `CROUCH`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `PAUSE`
- `SELECT`

### Key Binding Logic
- **Body Mode**: Based on **relative body position**, **body scale ratio**, **vertical velocity**, and **displacement**.
- **Hand Mode**: Based on **hand landmarks**, **hand position**, and **movement direction**.

## 📋 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MotionVerse
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Run the application:
   ```bash
   python app.py
   ```

2. **Calibrate**: Go to the **Calibration** tab to set your baseline body position.
3. **Select Mode**: Choose **Full-Body Mode** or **Hand Mode** on the control selection screen.
4. **Test**: Use the **Gesture Test** tab to verify your movements.
5. **Play**: Switch to the **Gaming** tab to control your game with body movements!

## 🎮 Game Configuration

Create game profiles in the **Settings** tab to customize:
- Game name
- Keyboard bindings for each action
- Jump thresholds
- Crouch thresholds
- Movement sensitivity

## 📋 Usage Examples

### Full-Body Mode
- Jump by performing a **deep squat** (triggers based on body height compression).
- Move left/right by **shifting your weight**.

### Hand Mode
- Use hand gestures to trigger **fast actions** or **UI navigation**.
- Ideal for games requiring quick, subtle inputs.

## 📂 Project Structure

```
MotionVerse/
├── app/
│   ├── core/
│   │   ├── calibration/      # Calibration logic and baseline management
│   │   ├── gestures/        # Gesture recognition modules
│   │   │   ├── body/         # Full-body gesture detection
│   │   │   ├── hand/         # Hand gesture detection
│   │   │   └── common/       # Shared action definitions and interfaces
│   │   ├── vision/           # Computer vision pipelines (OpenCV/MediaPipe)
│   │   └── input_dispatch/   # PyAutoGUI input simulation
│   ├── ui/
│   │   ├── main_window.py    # Main application window and UI structure
│   │   ├── dashboard_tab.py  # Dashboard UI
│   │   ├── control_mode_tab.py # Mode selection UI
│   │   ├── game_selection_tab.py # Game selection UI
│   │   ├── calibration_tab.py  # Calibration UI
│   │   ├── gesture_test_tab.py # Gesture testing UI
│   │   ├── gaming_tab.py     # Live gaming control UI
│   │   ├── analytics_tab.py  # Performance analytics UI
│   │   └── settings_tab.py   # Settings and configuration UI
│   ├── models/             # Model files (if any)
│   └── utils/                # Utility functions and helpers
├── scripts/                  # Helper scripts (tests, benchmarks)
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── config/                   # Configuration files
└── README.md
```

## 🧪 Testing

Run unit tests with pytest:

```bash
pytest tests/unit/test_gesture_engine.py
```

## 📦 Packaging

Create a standalone Windows executable:

```bash
pyinstaller --noconsole --onefile app.py
```

## 📝 Documentation

- [System Architecture](docs/architecture/system_architecture.md)
- [UI Specification](docs/ui/ui_specification.md)
- [Testing Strategy](docs/testing/testing_strategy.md)
- [Development Roadmap](docs/development/development_roadmap.md)

## 🤝 Contributing

Contributions are welcome! Please see our [Contribution Guidelines](docs/development/contributing.md) for details.

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
