import json
import os

notebook_dict = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project Evaluation: Old Photo Restoration\n",
    "In this notebook, we evaluate the performance of the generative image translation model used for restoring old photos.\n",
    "\n",
    "## 1. Metrics for Image Restoration\n",
    "Unlike classification tasks where we have clear \"Training Accuracy\" or \"Testing Accuracy\", generative image restoration tasks are evaluated using full-reference image quality metrics. We simulate a test set by taking clean, high-quality images, artificially degrading them (adding scratches, noise, and blur), passing them through the model, and then measuring how close the output is to the original clean ground truth.\n",
    "\n",
    "We use the following evaluation metrics:\n",
    "1. **MSE (Mean Squared Error)**: Measures the average squared difference between the estimated pixels and the actual image pixels. Lower is better.\n",
    "2. **PSNR (Peak Signal-to-Noise Ratio)**: Represents the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher is better.\n",
    "3. **SSIM (Structural Similarity Index Measure)**: Evaluates the visual impact of three characteristics of an image: luminance, contrast, and structure. Values range from -1 to 1, where 1 indicates perfect similarity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from skimage.metrics import mean_squared_error as mse\n",
    "from skimage.metrics import peak_signal_noise_ratio as psnr\n",
    "from skimage.metrics import structural_similarity as ssim\n",
    "import os\n",
    "import shutil\n",
    "from subprocess import call\n",
    "\n",
    "# Ensure we are in the correct directory\n",
    "if not os.path.exists('Global'):\n",
    "    print('Please run this notebook from the root directory of the repository.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Generating the Test Set\n",
    "We will load a clean image and synthetically degrade it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_synthetic_scratches(image_path, output_path):\n",
    "    img = cv2.imread(image_path)\n",
    "    if img is None:\n",
    "        raise ValueError(f\"Could not read {image_path}\")\n",
    "    \n",
    "    h, w = img.shape[:2]\n",
    "    degraded = img.copy()\n",
    "    \n",
    "    # Add random scratches\n",
    "    for _ in range(5):\n",
    "        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)\n",
    "        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)\n",
    "        thickness = np.random.randint(1, 4)\n",
    "        color = (200, 200, 200) # whitish scratch\n",
    "        cv2.line(degraded, (x1, y1), (x2, y2), color, thickness)\n",
    "    \n",
    "    # Add some gaussian noise\n",
    "    noise = np.random.normal(0, 15, degraded.shape).astype(np.int16)\n",
    "    degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)\n",
    "    \n",
    "    cv2.imwrite(output_path, degraded)\n",
    "    return img, degraded\n",
    "\n",
    "os.makedirs('eval_temp/input', exist_ok=True)\n",
    "os.makedirs('eval_temp/output', exist_ok=True)\n",
    "\n",
    "# We will use one of the test images as our \"clean\" base if available, \n",
    "# or just use an arbitrary one for demonstration.\n",
    "base_img_path = 'test_images/old/a.png'\n",
    "degraded_img_path = 'eval_temp/input/test_degraded.png'\n",
    "\n",
    "clean_img, degraded_img = add_synthetic_scratches(base_img_path, degraded_img_path)\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))\n",
    "ax1.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))\n",
    "ax1.set_title('Original Clean')\n",
    "ax2.imshow(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB))\n",
    "ax2.set_title('Synthetically Degraded')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Running the Model Inference\n",
    "We call the core restoration script to process our degraded test image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_restoration():\n",
    "    input_dir = os.path.abspath('eval_temp/input')\n",
    "    output_dir = os.path.abspath('eval_temp/output')\n",
    "    \n",
    "    # We invoke the provided python run script\n",
    "    cmd = f'python run.py --input_folder \"{input_dir}\" --output_folder \"{output_dir}\" --GPU -1 --with_scratch'\n",
    "    print(f\"Running command: {cmd}\")\n",
    "    call(cmd, shell=True)\n",
    "\n",
    "run_restoration()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Evaluation and Results\n",
    "Now we calculate the errors and testing scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "restored_path = 'eval_temp/output/final_output/test_degraded.png'\n",
    "if not os.path.exists(restored_path):\n",
    "    print(\"Restoration failed to produce output.\")\n",
    "else:\n",
    "    restored_img = cv2.imread(restored_path)\n",
    "    \n",
    "    # Resize restored image back to original shape if the model padded/resized it\n",
    "    h, w = clean_img.shape[:2]\n",
    "    restored_img = cv2.resize(restored_img, (w, h))\n",
    "    \n",
    "    # Metrics Calculation\n",
    "    # We convert to grayscale for SSIM as it's standard\n",
    "    clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)\n",
    "    restored_gray = cv2.cvtColor(restored_img, cv2.COLOR_BGR2GRAY)\n",
    "    degraded_gray = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    # Base Error (Degraded vs Clean)\n",
    "    base_mse = mse(clean_gray, degraded_gray)\n",
    "    base_psnr = psnr(clean_gray, degraded_gray)\n",
    "    base_ssim = ssim(clean_gray, degraded_gray, data_range=255)\n",
    "    \n",
    "    # Restored Error (Restored vs Clean)\n",
    "    res_mse = mse(clean_gray, restored_gray)\n",
    "    res_psnr = psnr(clean_gray, restored_gray)\n",
    "    res_ssim = ssim(clean_gray, restored_gray, data_range=255)\n",
    "    \n",
    "    print(\"=== TESTING SCORES & ERRORS ===\")\n",
    "    print(\"Before Restoration (Baseline Errors):\")\n",
    "    print(f\"MSE:  {base_mse:.2f}\")\n",
    "    print(f\"PSNR: {base_psnr:.2f} dB\")\n",
    "    print(f\"SSIM: {base_ssim:.4f}\\n\")\n",
    "    \n",
    "    print(\"After Restoration (Model Performance):\")\n",
    "    print(f\"MSE:  {res_mse:.2f} (Lower is better)\")\n",
    "    print(f\"PSNR: {res_psnr:.2f} dB (Higher is better)\")\n",
    "    print(f\"SSIM: {res_ssim:.4f} (Closer to 1.0 is better)\\n\")\n",
    "    \n",
    "    # Visualization\n",
    "    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))\n",
    "    ax1.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))\n",
    "    ax1.set_title('Original Clean')\n",
    "    ax2.imshow(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB))\n",
    "    ax2.set_title(f'Degraded\\nPSNR: {base_psnr:.2f}')\n",
    "    ax3.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))\n",
    "    ax3.set_title(f'Restored\\nPSNR: {res_psnr:.2f}')\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 }
}

with open('Evaluation.ipynb', 'w') as f:
    json.dump(notebook_dict, f, indent=1)
