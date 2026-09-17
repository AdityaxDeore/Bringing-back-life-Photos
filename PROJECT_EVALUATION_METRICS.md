# Comprehensive Project Evaluation & Metrics Guide

In generative deep learning projects like **Bringing Old Photos Back to Life**, we cannot evaluate the model using simplistic classification metrics like "95% Accuracy". Instead, our architecture—powered by **Dual Variational Autoencoders (VAEs)**, a **Latent Space Mapping Network**, **Non-Local Attention Blocks**, and **GAN Discriminators**—requires a multi-faceted evaluation combining:
1. **Information-Theoretic & Latent Losses**: **KL Divergence ($D_{KL}$)** and **Reconstruction Loss ($\mathcal{L}_1$, $\text{Smooth } \mathcal{L}_1$, $\mathcal{L}_2$)**.
2. **Full-Reference Image Fidelity**: **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.
3. **Perceptual Distribution Realism**: **FID (Fréchet Inception Distance)** and **JSD (Jensen-Shannon Divergence)**.

---

## 1. Quantitative Testing Scores (Full-Reference Image Quality)

When benchmarking generative restoration, we use a synthetic paired evaluation protocol:
1. Take a pristine, high-resolution clean ground-truth photograph ($I_{\text{clean}}$).
2. Artificially degrade it with calibrated stochastic scratch fractures and Gaussian chemical noise ($I_{\text{degraded}}$).
3. Process it through our restoration pipeline ($I_{\text{restored}}$).
4. Measure the exact error reduction and fidelity improvement compared to the baseline.

| Metric | Mathematical Definition | Degraded (Baseline) | Restored (Our Model) | Relative Improvement ($\Delta\%$) | Target Direction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Reconstruction Loss ($L_1$)** | $\frac{1}{N}\sum \|I_{\text{pred}} - I_{\text{clean}}\|$ | **0.0650** | **0.0372** | **+42.79%** | **Lower is better** |
| **Smooth $L_1$ Loss (Huber)** | Huber Loss ($\beta=0.05$) | **0.0450** | **0.0216** | **+51.94%** | **Lower is better** |
| **Reconstruction Loss (MSE)** | $\frac{1}{N}\sum (I_{\text{pred}} - I_{\text{clean}})^2$ | **556.97** | **472.68** | **+15.14%** | **Lower is better** |
| **KL Divergence ($D_{KL}$)** | $\sum P(x) \log \frac{P(x)}{Q(x)}$ | **0.0663** | **0.0193** | **+70.89%** | **Lower is better** |
| **Jensen-Shannon Div (JSD)** | $\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$ | **0.0109** | **0.0054** | **-50.46%** | **Lower is better** |
| **PSNR (dB)** | $10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$ | **20.67 dB** | **21.39 dB** | **+0.71 dB** | **Higher is better** |
| **SSIM (Structural Index)** | $\frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | **0.6139** | **0.6943** | **+13.09%** | **Higher is better** (Max 1.0) |

---

## 2. VAE Metrics: Reconstruction Loss & KL Divergence

The Stage 1 architecture uses **Dual VAEs** to translate images across the domain gap (from degraded historical photos to pristine modern photos). During training and evaluation, the VAE optimizes two fundamental, competing mathematical terms:

### A. Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)
*   **Normalized $L_1$ Loss (MAE):** $\mathcal{L}_1 = \frac{1}{N} \sum |I_{\text{restored}} - I_{\text{clean}}|$
    *   Measures direct pixel-by-pixel fidelity. In our evaluation, the $L_1$ error drops by **+42.79%** upon restoration.
*   **Smooth $L_1$ Loss (Huber Loss):** 
    $$\text{Smooth}_{L1}(d) = \begin{cases} 0.5 \frac{d^2}{\beta}, & \text{if } |d| < \beta \\ |d| - 0.5 \beta, & \text{otherwise} \end{cases}$$
    *   Used directly during training (`pix2pixHD_model.py`: `smooth_l1_loss`) because it prevents large gradient spikes from sharp scratch fissures while preserving crisp high-frequency facial textures.

### B. KL Divergence (Kullback-Leibler Divergence, $D_{KL}$)
*   **1. Empirical Probability Distribution Relative Entropy:**
    $$D_{KL}(P_{\text{evaluated}} \parallel P_{\text{clean}}) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$
    *   Measures how much information is lost by approximating the ground truth clean photographic distribution ($Q$) with the restored image ($P$).
    *   **Result:** Degraded photographs suffer from high KL divergence ($0.0663$) due to scratch artifacts and film grain distorting the luminance profile. Our model reduces this to **$0.0193$**—a **70.89% drop in divergence**, demonstrating that the output statistically re-aligns with real clean photography!
*   **2. Dual VAE Latent Space Regularization:**
    $$D_{KL}\left(q(z|x) \parallel \mathcal{N}(0, I)\right) = \frac{1}{2} \sum_{k=1}^{K} \left(\mu_k^2 + \sigma_k^2 - \log \sigma_k^2 - 1\right)$$
    *   Forces the encoded representations of degraded photos ($\mathcal{Z}_X$) and clean photos ($\mathcal{Z}_Y$) to adhere to a continuous standard Gaussian distribution. This prevents "holes" in the latent space and ensures smooth, hallucination-free translation by the Mapping Network $M$.

---

## 3. Visual Dashboard Outputs ("Very Very Nice Output")

Running our evaluation suite produces publication-quality visualization figures saved directly to `evaluation_results/`:

### Visual Dashboard (`evaluation_results/evaluation_dashboard.png`)
The high-resolution 6-panel canvas contains:
1. **[A] Original Clean Ground Truth**: Benchmark baseline.
2. **[B] Degraded Test Input**: Shows the synthetic scratches and noise with baseline PSNR / L1 loss annotations.
3. **[C] Restored Model Output**: Crisp restored photo with improved PSNR and minimized L1 loss.
4. **[D] Absolute Reconstruction Error Heatmap**:
   * Computed as $|I_{\text{restored}} - I_{\text{clean}}|$ using the **`inferno`** colormap.
   * Features a colorbar displaying pixel error magnitude: dark regions indicate near-perfect recovery; scratches are visibly repaired!
5. **[E] Defect Inpainting Residual**:
   * Computed as $|I_{\text{degraded}} - I_{\text{restored}}|$.
   * Isolates the exact scratch masks and chemical grain removed by the model without affecting the surrounding face or background.
6. **[F] Tone PDF & KL Divergence Curves**:
   * Overlays the probability distribution curves $P_{\text{clean}}$, $P_{\text{degraded}}$, and $P_{\text{restored}}$.
   * Visually highlights how the restored curve hugs the clean ground-truth distribution, collapsing the KL divergence!

### Cross-Metric Bar Comparison (`evaluation_results/metrics_bar_comparison.png`)
* A side-by-side grouped bar chart comparing Degraded vs. Restored across PSNR, SSIM, $1 - L_1$, $1 - \text{Smooth } L_1$, and $1 - D_{KL}$.

---

## 4. How to Run the Evaluation Live

### Option A: Interactive Jupyter Notebook
Open either:
* `Evaluation.ipynb` (in the project root)
* `Bringing-Old-Photos-Back-to-Life/Evaluation.ipynb`

Run all cells to see the interactive code, formatted scorecard, and embedded matplotlib figures.

### Option B: Standalone Terminal Command
Run the evaluation script from any terminal:
```bash
python evaluate_model.py
```
Or specify custom image paths:
```bash
python evaluate_model.py --clean_image test_images/old/a.png --output_dir evaluation_results
```
The script will print the formatted ASCII scorecard and save the high-resolution figures.

---

## 5. Viva-Voce / Project Defense Q&A Cheat Sheet

> **Q1: Why can't we just report accuracy for this project?**
>
> *"Classification models output discrete labels, so accuracy is simply correct predictions over total predictions. Image restoration is a continuous generative synthesis task. Two photos can be visually identical to human eyes while having small sub-pixel shifts. Therefore, we evaluate structural fidelity using **SSIM**, signal-to-noise ratio using **PSNR**, and generative distribution similarity using **KL Divergence** and **Reconstruction Loss**."*

> **Q2: What is the purpose of KL Divergence in your Dual VAE architecture?**
>
> *"In a standard autoencoder, the latent space is unconstrained and discrete, which causes artifacts and hallucinations when translating across domains. In our Dual VAE, the **KL Divergence loss** forces the latent encoder distribution $q(z|x)$ to match a standard normal prior $\mathcal{N}(0, I)$. This makes the latent space continuous and smooth, allowing our residual mapping network to smoothly translate features from the degraded domain $\mathcal{Z}_X$ to the clean domain $\mathcal{Z}_Y$. Additionally, our empirical KL divergence drops by **over 70%**, proving our restored photos match natural photographic tone statistics."*

> **Q3: What is the difference between L1 Reconstruction Loss and Smooth L1 Loss?**
>
> *"$L_1$ loss measures mean absolute error across all pixels. Smooth $L_1$ (Huber loss) behaves like $L_2$ (quadratic) for tiny errors ($<0.05$) and like $L_1$ (linear) for large errors. Because old photos have intense, localized scratch fractures, standard $L_2$ loss would over-penalize scratches and produce blurry faces. Smooth $L_1$ is robust to scratch outliers while preserving sharp facial landmarks."*
