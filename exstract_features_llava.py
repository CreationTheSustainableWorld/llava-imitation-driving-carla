import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import warnings

# === 設定 ===
model_id = "llava-hf/llava-1.5-7b-hf"
image_dir = "_out"  # 画像ディレクトリ
feature_dir = "features_llava"  # 特徴保存先
prompt = "<|user|>\n<image>\nYou are driving cautiously in a crowded urban area. What should you do?\n<|end|>\n<|assistant|>"

os.makedirs(feature_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === モデルとプロセッサの読み込み ===
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
model.eval()

# === 特徴抽出ループ ===
with torch.no_grad():
    for fname in tqdm(sorted(os.listdir(image_dir))):
        if not fname.endswith(".png"):
            continue

        # 入力準備
        try:
            image = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            warnings.warn(f"❌ 画像読み込み失敗: {fname} をスキップします: {e}")
            continue

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)

        # 中間特徴の抽出（最後のhidden_stateの[CLS]トークン）
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True
        )
        cls_feature = outputs.hidden_states[-1][:, 0]  # shape: (1, hidden_dim)

        # 保存
        out_path = os.path.join(feature_dir, fname.replace(".png", ".pt"))
        torch.save(cls_feature.squeeze(0).cpu(), out_path)
