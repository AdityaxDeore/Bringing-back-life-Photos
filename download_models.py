import os
import sys
import subprocess
import shutil

print("=== Downloading Pretrained Models & Checkpoints ===")

# 1. Download shape_predictor_68_face_landmarks.dat if missing
landmark_path = "Face_Detection/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(landmark_path):
    print("Downloading 68-face landmarks predictor...")
    bz2_path = landmark_path + ".bz2"
    import urllib.request
    import bz2
    urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", bz2_path)
    with open(landmark_path, "wb") as f_out, bz2.BZ2File(bz2_path, "rb") as f_in:
        f_out.write(f_in.read())
    if os.path.exists(bz2_path):
        os.remove(bz2_path)
    print("Landmarks extracted successfully.")
else:
    print("Landmarks model already present.")

# 2. Check Face Enhancement & Global Checkpoints
face_ckpt = "Face_Enhancement/checkpoints/Setting_9_epoch_100/latest_net_G.pth"
global_ckpt = "Global/checkpoints/detection/FT_Epoch_latest.pt"

if not os.path.exists(face_ckpt) or not os.path.exists(global_ckpt):
    print("Cloning checkpoints from official Hugging Face mirror (fast & reliable)...")
    temp_dir = "temp_hf_checkpoints"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    subprocess.run(["git", "clone", "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life", temp_dir], check=True)
    
    os.makedirs("Face_Enhancement/checkpoints", exist_ok=True)
    os.makedirs("Global/checkpoints", exist_ok=True)
    
    shutil.copytree(os.path.join(temp_dir, "Face_Enhancement", "checkpoints"), "Face_Enhancement/checkpoints", dirs_exist_ok=True)
    shutil.copytree(os.path.join(temp_dir, "Global", "checkpoints"), "Global/checkpoints", dirs_exist_ok=True)
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Checkpoints successfully downloaded and moved into place.")
else:
    print("All face and global checkpoints are already present.")

print("Pretrained models ready!")
