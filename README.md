# 🚗 LLaVA-based Autonomous Driving Agent in CARLA

This project demonstrates a simple imitation learning pipeline using [LLaVA 1.5](https://github.com/haotian-liu/LLaVA) (a vision-language model) combined with a custom-trained MLP to control a vehicle in the CARLA simulator.

![HUD Screenshot](images/hud_screenshot.png)

---

## 🎯 Overview

- Vision model: **llava-hf/llava-1.5-7b-hf**  
- Action model: **Custom MLP (linear layers)**  
- Input: Image + Prompt  
- Output: Continuous actions: **steer, throttle, brake**

The system observes RGB camera images in a simulated environment and makes real-time driving decisions based on visual and contextual information.

---

## 🧠 Model Architecture

```text
Camera Image + Prompt
        ↓
    LLaVA (frozen)
        ↓
Last hidden state [CLS] (dim=4096)
        ↓
   MLP Head (4096 → 512 → 3)
        ↓
[Steer, Throttle, Brake]
```

---

## 🖼️ Demo

![Frame Sample](images/example_frame.png)

---

## 🛠️ Requirements

```bash
pip install torch transformers carla pygame Pillow numpy
```

---

## 🚀 How to Run

1. Launch CARLA simulator (`CarlaUE4.exe`)
2. Then run:

```bash
python run_carla_imitation_agent.py
```

> You should see the vehicle driving based on visual input and the HUD displaying its control states.

---

## 📁 Project Structure

```text
.
├── run_carla_imitation_agent.py       # Main execution script
├── model/
│   ├── mlp_action_head.pth            # Trained MLP
│   └── llava_config/                  # (optional) LLaVA model weights
├── images/
│   ├── hud_screenshot.png             # HUD screenshot
│   └── example_frame.png              # Camera image sample
├── data/
│   ├── actions.json                   # Saved control logs
│   └── log_video.mp4                  # (optional) driving video
└── README.md
```

---

## 🤖 Training Summary

- Training data was collected from a behavior agent in CARLA
- Each image was paired with its corresponding `[steer, throttle, brake]` vector
- LLaVA was frozen, and only the MLP was trained

---

## 📌 Note

This is a lightweight imitation learning demonstration. You can:
- Improve precision by fine-tuning LLaVA
- Add multi-camera views
- Incorporate memory or language-based reasoning

---

## 👤 Author

**Daiki Matsuba**  
- GitHub: [CreationTheSustainableWorld](https://github.com/CreationTheSustainableWorld)  
- Portfolio: [Google Sites](https://sites.google.com/view/job-application-portfolio)
