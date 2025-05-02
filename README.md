# 🚗 LLaVA-based Autonomous Driving Agent in CARLA

This project implements a vision-language imitation learning agent in CARLA using LLaVA and a custom MLP action head.

## 🎯 Features
- 🔍 Vision encoder: LLaVA 1.5 (7B)
- 🤖 Action head: Custom MLP trained on real CARLA driving data
- 🎮 Real-time control in CARLA
- 📺 HUD and camera view rendered with pygame
- 📦 All models and scripts are ready to run

## 🖼️ Sample Output
![HUD View](images/hud_screenshot.png)

## 🧠 Model Architecture
LLaVA (image + prompt) → last hidden state → MLP → [steer, throttle, brake]

## 🛠️ Requirements
```bash
pip install -r requirements.txt
