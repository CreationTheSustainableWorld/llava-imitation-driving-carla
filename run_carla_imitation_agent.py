import carla
import torch
import numpy as np
import pygame
import time
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration

# === MLPヘッドモデルの定義 ===
class ActionHead(torch.nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)

# === 初期設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt = "<|user|>\n<image>\nYou are driving cautiously in a crowded urban area. What should you do?\n<|end|>\n<|assistant|>"
WIDTH, HEIGHT = 1280, 720

# === モデルとプロセッサのロード ===
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
llava_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
).to(device).eval()
mlp_model = ActionHead(input_dim=4096).to(device)
mlp_model.load_state_dict(torch.load("trained_models/mlp_action_head.pth"))
mlp_model.eval()

# === pygame初期化 ===
pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA LLaVA Agent Viewer")
font = pygame.font.Font(None, 28)

# === CARLAに接続 ===
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# === 車両生成 ===
blueprint_library = world.get_blueprint_library()
bp = blueprint_library.filter("vehicle.*")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.try_spawn_actor(bp, spawn_point)

# === カメラセンサ設定 ===
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# === 最新画像を保存するバッファ ===
latest_image = {"array": None}

def process_and_control(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    pil_img = Image.fromarray(array)
    latest_image["array"] = array

    inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        outputs = llava_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True
        )
        cls_feature = outputs.hidden_states[-1][:, 0].float()
        action = mlp_model(cls_feature).squeeze(0).cpu().numpy()

    steer = float(np.clip(action[0], -1, 1))
    throttle = float(np.clip(action[1], 0, 1))
    brake = float(np.clip(action[2], 0, 1))

    control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
    vehicle.apply_control(control)

    return steer, throttle, brake

camera.listen(lambda image: process_and_control(image))

# === メインループ ===
clock = pygame.time.Clock()
try:
    while True:
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt()

        display.fill((0, 0, 0))

        # === 最新画像表示 ===
        if latest_image["array"] is not None:
            surface = pygame.surfarray.make_surface(latest_image["array"].swapaxes(0, 1))
            display.blit(surface, (0, 0))

            velocity = vehicle.get_velocity()
            speed = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
            control = vehicle.get_control()

            # === HUD描画 ===
            texts = [
                f"Speed: {speed:.1f} km/h",
                f"Steer: {control.steer:.2f}",
                f"Throttle: {control.throttle:.2f}",
                f"Brake: {control.brake:.2f}"
            ]
            for i, text in enumerate(texts):
                txt_surface = font.render(text, True, (255, 255, 255))
                display.blit(txt_surface, (10, 10 + i * 25))

        pygame.display.flip()
except KeyboardInterrupt:
    print("停止中...")
finally:
    camera.stop()
    vehicle.destroy()
    pygame.quit()
