import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

# === 共通設定 ===
model_id = "llava-hf/llava-1.5-7b-hf"
save_dir_base = "./saved_llava_models"
os.makedirs(save_dir_base, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === original モデル保存済みのため省略 ===

# === delete モデル（中間特徴抽出まで） ===
model_delete = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
save_path_delete = os.path.join(save_dir_base, "llava-1.5-7b-hf_delete")
model_delete.save_pretrained(save_path_delete)
print(f"✅ Saved delete model to {save_path_delete}")

# === addition モデル（中間特徴→行動出力） ===
class LlavaWithActionHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        # 🔧 LLaVAの中のLLaMAのhidden sizeを取得
        hidden_size = self.base.language_model.config.hidden_size
        hidden_size = self.base.language_model.config.hidden_size


        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # 出力: steer, throttle, brake
        )

    def forward(self, input_ids, attention_mask=None, pixel_values=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        # 🔽 decoderの最終層の[CLS]トークンを利用
        hidden_state = outputs.hidden_states[-1][:, 0]  # shape: (B, hidden)

        return self.action_head(hidden_state)

model_base = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
model_add = LlavaWithActionHead(model_base).to(device)

save_path_add = os.path.join(save_dir_base, "llava-1.5-7b-hf_addition")
os.makedirs(save_path_add, exist_ok=True)
torch.save(model_add.state_dict(), os.path.join(save_path_add, "pytorch_model.bin"))
print(f"✅ Saved addition model to {save_path_add}")
