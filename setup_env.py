import os
import sys
import subprocess
import shutil

print("=== Setting up Bringing Old Photos Back to Life Environment ===")

# Step 1: Install Python dependencies
print("\n[1/4] Installing Python dependencies from requirements.txt...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# Step 2: Set up Synchronized-BatchNorm
print("\n[2/4] Setting up Synchronized-BatchNorm submodules...")
submodules = [
    ("Face_Enhancement/models/networks", "Face_Enhancement/models/networks/sync_batchnorm"),
    ("Global/detection_models", "Global/detection_models/sync_batchnorm")
]

for parent_dir, target_dir in submodules:
    if not os.path.exists(target_dir):
        print(f"Cloning Synchronized-BatchNorm into {parent_dir}...")
        repo_dir = os.path.join(parent_dir, "Synchronized-BatchNorm-PyTorch")
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", "https://github.com/vacancy/Synchronized-BatchNorm-PyTorch", repo_dir], check=True)
        src = os.path.join(repo_dir, "sync_batchnorm")
        if os.path.exists(src):
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
        print(f"Synchronized-BatchNorm configured at {target_dir}")
    else:
        print(f"Synchronized-BatchNorm already present at {target_dir}")

# Step 3: Run model downloader if checkpoints missing
print("\n[3/4] Checking required pretrained weights & landmarks...")
checkpoint_paths = [
    "Face_Detection/shape_predictor_68_face_landmarks.dat",
    "Face_Enhancement/checkpoints/Setting_9_epoch_100/latest_net_G.pth",
    "Global/checkpoints/detection/FT_Epoch_latest.pt"
]

missing = [p for p in checkpoint_paths if not os.path.exists(p)]
if missing:
    print(f"Missing checkpoint files: {missing}")
    print("Downloading weights via download_models.py...")
    subprocess.run([sys.executable, "download_models.py"], check=True)
else:
    print("All pretrained checkpoints and landmark models are verified and present!")

print("\n[4/4] Environment setup completed successfully!")
print("You can now run the GUI using: python GUI.py")
print("Or run evaluation using: jupyter notebook Evaluation.ipynb")
