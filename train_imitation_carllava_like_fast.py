import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

# === パス設定 ===
FEATURE_DIR = "features_llava"
ACTION_FILE = "_out/actions.json"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

# === データセット定義 ===
class FeatureActionDataset(Dataset):
    def __init__(self, feature_dir, action_file):
        self.feature_dir = feature_dir
        self.samples = []
        missing = 0
        total = 0
        with open(action_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                for fname, action in obj.items():
                    total += 1
                    pt_path = os.path.join(feature_dir, fname.replace(".png", ".pt"))
                    if os.path.exists(pt_path):
                        self.samples.append((pt_path, action))
                    else:
                        missing += 1
                        print(f"❌ .ptファイルが存在しません: {pt_path}")
        print(f"✅ 有効なデータ数: {len(self.samples)} / 全データ数: {total}（欠損: {missing}）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, action = self.samples[idx]
        feature = torch.load(pt_path).half()  # 🔧 float16に統一
        action_tensor = torch.tensor([
            action["steer"],
            action["throttle"],
            action["brake"]
        ], dtype=torch.float16)  # 🔧 こちらも float16 に
        return feature, action_tensor

# === MLPモデル定義 ===
class ActionHead(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)

# === 学習処理 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FeatureActionDataset(FEATURE_DIR, ACTION_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ActionHead().to(device).float()  # 🔧 モデルも float16 に変換
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for features, actions in loader:
        features = features.to(device).float()
        actions = actions.to(device).float()

        preds = model(features)
        loss = criterion(preds, actions)

        # === デバッグ出力 ===
        print("🟡 --- Debug Info ---")
        print("features dtype:", features.dtype)
        print("actions dtype:", actions.dtype)
        print("preds:", preds)
        print("actions:", actions)
        print("diff:", preds - actions)
        print("loss:", loss.item())
        print("preds min:", preds.min().item(), "max:", preds.max().item())
        print("actions min:", actions.min().item(), "max:", actions.max().item())
        print("----------------------")

        if not torch.isfinite(loss):
            print("❌ 非数 (NaN or Inf) の loss が検出されました。学習を中断します。")
            exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# モデル保存
os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), "trained_models/mlp_action_head.pth")
print("✅ MLPヘッドモデルを保存しました: trained_models/mlp_action_head.pth")
