# Bringing Old Photos Back to Life (Photo Restoration)

**Creator / Author:** Aditya Deore  
**Repository:** [https://github.com/AdityaxDeore/Bringing-back-life-Photos.git](https://github.com/AdityaxDeore/Bringing-back-life-Photos.git)

An AI-powered deep latent space translation framework designed to restore vintage, faded, degraded, and physically scratched historical photographs. Includes an interactive GUI, automated end-to-end setup scripts, and a performance evaluation notebook (`Evaluation.ipynb`).

---

## ⚡ Quick Start for Users & AI Agents

If you just cloned this repository or gave this project to an AI agent to execute, run the automated setup script first:

### Step 1: Automated Installation & Dependency Setup
Run the setup script which clones required submodules, installs all Python requirements, and downloads required pretrained models / landmark weights:

```bash
python setup_env.py
```

*Or manually in PowerShell:*
```powershell
pip install -r requirements.txt
python download_models.py
```

---

## 🖥️ Graphical User Interface (GUI)

An interactive, user-friendly desktop application is included where you can pick any photo, process it in one click, and see the before-and-after results side-by-side:

```bash
python GUI.py
```

### How to Use the GUI:
1. Run `python GUI.py`.
2. Click **Browse** and select any photo (e.g. from `test_images/old_w_scratch/` or any image on your PC).
3. Click **Restore Photo**.
4. The backend neural networks will run Stage 1 (Overall Restoration), Stage 2 (Face Detection), Stage 3 (Face Detail Enhancement), and Stage 4 (Blending).
5. The restored photo will display immediately in the GUI and automatically save.

---

## 🚀 Command Line Usage

You can run batch restoration directly from the terminal.

### 1. Photos with Scratches & Severe Degradation
```bash
python run.py --input_folder test_images/old_w_scratch --output_folder output_scratch --GPU -1 --with_scratch
```

### 2. Photos without Physical Scratches (Enhancement Only)
```bash
python run.py --input_folder test_images/old --output_folder output_clean --GPU -1
```

### 3. High-Resolution (HR) Mode
```bash
python run.py --input_folder test_images/old_w_scratch --output_folder output_hr --GPU -1 --with_scratch --HR
```

> **Note on GPU**: Set `--GPU 0` if you have an NVIDIA CUDA GPU, or `--GPU -1` to run reliably on CPU.

---

## 📊 Performance & Academic Evaluation (`Evaluation.ipynb`)

For academic assignments, reports, and quantitative benchmarking:
Open and run **`Evaluation.ipynb`** in Jupyter Notebook or VS Code.

The notebook computes:
- **MSE (Mean Squared Error)**: Pixel reconstruction discrepancy.
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantitative signal quality metric (dB).
- **SSIM (Structural Similarity Index Measure)**: Preservation of structural, luminance, and contrast details.
- Side-by-side visual comparisons between Original Clean, Degraded, and Restored outputs.

To launch the evaluation:
```bash
jupyter notebook Evaluation.ipynb
```

---

## 📁 Dataset & Sample Images

Sample datasets are pre-included in the repository for immediate testing:
- **`test_images/old_w_scratch/`**: Real historical photographs with deep scratches, creases, and abrasions.
- **`test_images/old/`**: Vintage degraded photographs without physical tears.
- **`training_data/Real_L_old/` & `training_data/Real_RGB_old/`**: Baseline domain data directories for feature mapping.

---

## 🛠️ Architecture & Pipeline Overview

1. **Triplet Domain Translation**: Transforms unstructured degradation into latent representations via dedicated VAE networks.
2. **Scratch Detection & Inpainting**: Non-local attention networks detect and fill irregular cracks and scratches.
3. **Progressive Face Enhancement (SPADE)**: Localizes facial landmarks via `dlib`, scales facial features, and enhances facial realism using a generative progressive network.
4. **Poisson & Blur Blending**: Restores facial crops back to the global context with seamless boundary transitions.

---

## 📄 License & Conduct
Developed and maintained by **Aditya Deore**. Released under the MIT License.
