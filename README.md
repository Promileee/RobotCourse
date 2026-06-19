# Intelligent Robot Challenging Practice — Vision-Guided Laser Pointing System

> [中文版本 (Chinese Version)](README_cn.md)

This repository contains all program code for the **Intelligent Robot Challenging Practice** course experiment. The project builds a **vision-guided laser pointing system** that uses an RGB-D camera as the sensing element and an Arduino-driven 2-axis gimbal as the actuator, directing a laser spot to sequentially point at corresponding cylindrical materials and square targets in the color sequence specified by a QR code.

## System Architecture

A **host PC (Python) + microcontroller (Arduino)** two-tier architecture:

- **Host PC**: Handles all vision algorithms and high-level decision-making, sending commands to Arduino via USB virtual serial port (9600 bps)
- **Microcontroller (Arduino)**: Drives two stepper motors (16 micro-step subdivision, 3200 steps/rev) controlling the gimbal's Pan (horizontal) and Tilt (pitch) axes
- **Camera**: Orbbec RGB-D camera, decoupled as a background service process via ZeroMQ PUB-SUB pattern, shared by multiple consumer programs

## Project Structure

### Camera & Data Infrastructure

| File | Description |
| ---- | ----------- |
| `camera_manager.py` | Unified camera manager (ZMQ client), subscribes to RGB + depth data published by the server |
| `server_live.py` | Real-time hardware server, connects to physical cameras and broadcasts data |
| `server_playback.py` | Offline playback server, reads locally recorded data and emulates a live stream |
| `record_data_v1.py` | Data recording utility, saves RGB frames as AVI and depth frames as NPZ |

### Vision Recognition Modules

| File | Description |
| ---- | ----------- |
| `QR_code_recognize_v1.py` | QR code recognition using OpenCV QRCodeDetector, parses color execution sequence (1=Red, 2=Green, 3=Blue) |
| `rect_recognize.py` | Cylindrical material detection via HSV color space segmentation + depth information joint verification, detects red/green/blue cylinders |
| `target_recognize.py` | Target recognition base class, three-stage tracking pipeline using ORB feature matching + rigid-body geometric constraints (square / equilateral triangle) |
| `target_recognize_v1.py` | Target recognition V1, adds temporal filtering (EMA smoothing + outlier frame rejection) on top of the base class |
| `target_recognize_v2.py` | Target recognition V2, adds sliding-window median filter + constant-velocity Kalman filter to smooth ORB mismatching flyers |
| `laser_recognize.py` | Laser spot detection base class, using grayscale brightness threshold + depth distance filtering |
| `laser_recognize_v3.py` | Laser detection V3, adds depth neighborhood search (compensating RGB-D parallax), dual-channel joint brightness verification (grayscale + HSV-V), multi-candidate association tracking with scoring |

### Control Modules

| File | Description |
| ---- | ----------- |
| `motor_control.py` | Motor serial communication module, supports three modes: homing reset (Mode 0), open-loop position control (Mode 1), closed-loop velocity control (Mode 2) |
| `open_loop_control_v2.py` | Open-loop controller, uniform grid sampling in angle space → MLP neural network training (2×32×16×2) → establishes nonlinear pixel-to-motor-angle mapping |
| `close_loop_control_v2.py` | Closed-loop controller V2 (class-based), PD velocity servo + acceleration limiting + stability判定, supports flexible color specification |

### Mission Scripts (Main Programs)

| File | Description |
| ---- | ----------- |
| **`mission_open_close_v1.py`** | **Current main program** — hybrid strategy: pure open-loop for materials, "open-loop coarse positioning + closed-loop PD fine locking" two-stage for targets |
| `mission_open_close.py` | Initial hybrid version (same core logic as V1; V1 is the optimized successor) |
| `mission_open_loop_v3.py` | Pure open-loop V3: open-loop for materials + continuous tracking then open-loop pointing for targets |
| `mission_open_loop_v2.py` | Pure open-loop V2: adds loop execution and motor reset |
| `mission_open_loop.py` | Pure open-loop initial version |

### Debugging & Utilities

| Path | Description |
| ---- | ----------- |
| `debug/` | Debug scripts, historical version backups, recorded data samples, Gemini-assisted experiment scripts, test screenshots |
| `mission.md` | Brief project task description |

## Core Workflow (mission_open_close_v1.py)

```
Initialization Phase (once):
  1. Cylindrical material coordinate detection (before any motor movement,
     leveraging the fixed-position assumption of materials)
  2. Motor serial connection & confirmation
  3. Target & laser recognition module initialization
     (auto-compute laser search ROI from target spatial positions)
  4. Uniform grid sampling in angle space → MLP neural network training
     (pixel → angle mapping)

Cyclic Execution Phase (each round):
  1. Scan QR code → obtain color sequence (e.g., Red → Green → Blue)
  2. For each color in sequence:
     a. Point to material (pure open-loop NN prediction + set_position)
     b. Re-identify target (from scratch)
     c. Coarse open-loop target positioning (NN prediction + set_position)
     d. Re-identify target (adapt to new viewpoint)
     e. Closed-loop PD fine locking (visual feedback velocity servo,
        success when laser stays within dead zone for 15 consecutive frames)
  3. Motor reset after all colors, wait for next round
```

## Control Strategy

- **Material pointing**: Pure open-loop — materials are large with loose pointing precision requirements; laser spot detection is difficult outside the target disk region
- **Target pointing**: Open-loop + Closed-loop hybrid — open-loop handles large-range rapid relocation near the target; closed-loop PD handles local pixel-level fine locking
- **PD parameters**: Kp=0.15, Kd=0.06, dead zone=5 px, max speed=200 °/s, max acceleration=300 °/s², closed-loop timeout=10 s

## How to Run

```bash
# 1. Start the camera server first (choose one)
python server_live.py       # Connect to physical cameras
python server_playback.py   # Play back offline recorded data

# 2. Run the main program
python mission_open_close_v1.py
```

## Team

- **Wang Renjia (王仁嘉)**: Camera data transmission architecture, target recognition & tracking algorithms, laser spot detection algorithms, open-loop calibration & neural network mapping, host PC main program framework
- **Jiang Weiyang (蒋维阳)**: Hardware setup & debugging, QR code recognition module, Arduino stepper motor control firmware, closed-loop PD velocity control algorithm design & parameter tuning

## Experiment Report

See `实验报告.docx` (Intelligent Robot Challenging Practice — Experiment Summary Report) for complete algorithm derivations, parameter tuning rationale, and quantitative performance metrics.
