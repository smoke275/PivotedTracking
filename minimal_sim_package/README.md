# Minimal Simulation Package

This is a self-contained package for running a minimal agent simulation.

## 📦 Package Contents

```
minimal_sim_package/
├── simple_sim_minimal.py              ← Main simulation file (run this!)
├── README.md                          ← This file
│
└── multitrack/                        ← Required modules
    ├── __init__.py
    │
    ├── models/
    │   ├── __init__.py
    │   ├── simulation_environment.py  ← Environment with walls/doors
    │   │
    │   └── agents/
    │       ├── __init__.py
    │       └── visitor_agent.py       ← Agent movement model
    │
    └── utils/
        ├── __init__.py
        └── config.py                  ← Configuration constants
```

## 🚀 How to Use

### 1. Install Dependencies
```bash
pip install pygame numpy
```

### 2. Run the Simulation
```bash
python simple_sim_minimal.py
```

## 🎮 Controls

- **Arrow Keys**: Control the agent
  - Up/Down: Move forward/backward
  - Left/Right: Rotate left/right
- **I**: Toggle info display
- **ESC**: Quit simulation

## 📝 Features

- Simple unicycle agent movement
- Indoor environment with walls and doors
- Real-time position and heading display
- Collision detection with walls
- Door pass-through capability

## 🔧 Customization

You can easily modify:
- **Agent speed**: Edit `LEADER_LINEAR_VEL` and `LEADER_ANGULAR_VEL` in `multitrack/utils/config.py`
- **Starting position**: Change `initial_position` in `simple_sim_minimal.py` (line 56)
- **Environment layout**: Modify walls/doors in `multitrack/models/simulation_environment.py`
- **Agent appearance**: Change `AGENT_COLOR` and `AGENT_RADIUS` in `simple_sim_minimal.py`

## 📋 File Descriptions

- **simple_sim_minimal.py**: Main simulation loop, handles pygame rendering and user input
- **simulation_environment.py**: Defines the environment (walls, doors, windows)
- **visitor_agent.py**: Implements the unicycle motion model with collision handling
- **config.py**: Central configuration for colors, speeds, and other constants

## 🆕 Starting Fresh

This package is designed to be completely standalone. Just copy the entire `minimal_sim_package` folder anywhere and run it!

No other files from the original project are needed.

---

**Enjoy your minimal simulation!** 🎉
