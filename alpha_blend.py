import os
from PIL import Image
import numpy as np

# 參數設定
alpha = 0.2  # 混合比例，0 = 原圖，1 = 完全 fake_B
image_dir = './results/leica_gbw/test_latest/images'
output_dir = './results/leica_gbw/test_latest/images/alpha_blended'

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 找出所有 _real_A.png 檔案
image_files = [f for f in os.listdir(image_dir) if f.endswith('_real_A.png')]

for real_name in image_files:
    base_name = real_name.replace('_real_A.png', '')
    fake_name = f"{base_name}_fake_B.png"

    real_path = os.path.join(image_dir, real_name)
    fake_path = os.path.join(image_dir, fake_name)

    if not os.path.exists(fake_path):
        print(f"Missing: {fake_path}, skipping.")
        continue

    real_img = Image.open(real_path).convert('RGB')
    fake_img = Image.open(fake_path).convert('RGB')

    # 確保兩圖尺寸一致
    if real_img.size != fake_img.size:
        fake_img = fake_img.resize(real_img.size)

    # 進行 alpha blending
    real_np = np.array(real_img).astype(np.float32)
    fake_np = np.array(fake_img).astype(np.float32)

    blended_np = (1 - alpha) * real_np + alpha * fake_np
    blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

    # 儲存結果
    blended_img = Image.fromarray(blended_np)
    save_path = os.path.join(output_dir, f"{base_name}_blended_{alpha:.2f}.png")
    blended_img.save(save_path)

    print(f"Saved: {save_path}")

print("✅ Blending done.")

