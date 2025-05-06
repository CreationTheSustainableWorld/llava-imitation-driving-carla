# 🚗 LLaVA-Based Vision-to-Action Imitation Model for Autonomous Driving

This project demonstrates how to repurpose the [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) multimodal model for **real-time driving control** using **CARLA simulator**.  
By extracting intermediate visual features from LLaVA and training a lightweight MLP, we achieve image → action mapping for autonomous vehicles.

---

## 📌 Overview

```
Image (RGB) + Prompt
     ↓
[LLaVA: vision encoder + language decoder]
     ↓ (CLS token)
Intermediate Visual Feature
     ↓
[MLP Head]
     ↓
Action: steer, throttle, brake
```

---

## ✅ Key Features

- 🔄 **Modular design**: Feature extraction, training, and inference are fully decoupled.
- 🎯 **Practical structure**: End-to-end flow from CARLA simulation → data collection → training → real-time control.
- 🧠 **Creative adaptation**: LLaVA is used not for text generation but for decision-making features.
- 🧪 **Tested in CARLA**: Live vehicle control via real-time inference.

---

## 🛠️ Folder Structure

```
.
├── data_collection/              # CARLA agent script for image-action collection
├── features_extraction/         # Extract LLaVA intermediate features (.pt)
├── mlp_training/                # Train MLP from features to actions
├── real_time_control/           # Inference script for CARLA real-time driving
├── trained_models/              # Saved MLP model
├── _out/                        # Collected images and actions
└── README.md
```

---

## 📂 Components

| Folder | Description |
|--------|-------------|
| `data_collection/` | Records driving images + actions from CARLA |
| `features_extraction/` | Uses LLaVA to extract CLS token from each image |
| `mlp_training/` | Trains MLP (ActionHead) on features-action pairs |
| `real_time_control/` | Loads LLaVA + MLP and controls CARLA vehicle in real time |

---

## 📈 Improvements & Extensions

| Challenge | Future Direction |
|----------|------------------|
| Only MLP | Use Transformer or CNN-based ActionHead |
| No temporal context | Add LSTM/GRU for time-series modeling |
| No reward usage | Apply RL fine-tuning (e.g., PPO, TD3+BC) |
| Limited visualization | Add training/inference visualization dashboards |

---

## 📽️ Demo

_🧩 Coming Soon: Demo video of agent driving in CARLA._

---

## 📚 References

- [LLaVA Model (Hugging Face)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [CARLA Simulator](https://carla.org/)
- [TD3+BC: Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860)

---

## 👤 Author

GitHub: [CreationTheSustainableWorld](https://github.com/CreationTheSustainableWorld)
