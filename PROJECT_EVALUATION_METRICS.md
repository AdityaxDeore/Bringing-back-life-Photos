# Project Evaluation & Metrics Guide

In generative deep learning projects like **Bringing Old Photos Back to Life**, we cannot evaluate the model using simple metrics like "95% Accuracy" (which is used for classification tasks). Instead, we must evaluate both the **structural accuracy** and the **perceptual realism** of the generated images.

Here is a comprehensive breakdown of the evaluations, training/testing scores, and specific VAE/GAN metrics to include in your presentation.

---

## 1. Testing Scores (Quantitative Image Metrics)

When testing the model, we use a synthetic dataset: we take a pristine, high-resolution photo, artificially degrade it (add scratches, noise, blur), run it through our model, and compare the output to the original pristine photo.

You can demonstrate this live using the `Evaluation.ipynb` script provided in the repository.

*   **PSNR (Peak Signal-to-Noise Ratio):** 
    *   **What it is:** Measures the ratio of the maximum possible pixel value to the power of the distorting noise (error).
    *   **Score Interpretation:** **Higher is better.** (Usually measured in dB. Scores above 25-30 dB are considered very good).
*   **SSIM (Structural Similarity Index):** 
    *   **What it is:** Evaluates the visual impact of luminance, contrast, and structure, mimicking how the human eye perceives image quality.
    *   **Score Interpretation:** **Closer to 1.0 is better.** (1.0 means perfect structural match).
*   **MSE (Mean Squared Error):**
    *   **What it is:** The mathematical average of the squared differences between the generated pixels and the original pixels.
    *   **Score Interpretation:** **Lower is better.**

---

## 2. GAN Evaluations (Generative Adversarial Network)

Because PSNR and SSIM only measure pixel-by-pixel differences, they often penalize GANs for generating highly realistic but slightly shifted textures (e.g., realistic skin pores that don't perfectly align with the original). Therefore, we use perceptual metrics.

*   **FID (Fréchet Inception Distance):**
    *   **What it is:** The gold standard for GAN evaluation. It compares the distribution of generated images with the distribution of real images at a deep feature level using a pre-trained Inception-v3 network.
    *   **Score Interpretation:** **Lower is better.** A lower FID means the generated photos look statistically identical to real human photos.
*   **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **What it is:** Measures "perceptual similarity." It judges how similar two images look to a human rather than relying on strict mathematical pixel differences.
    *   **Score Interpretation:** **Lower is better.**

---

## 3. VAE Evaluations (Variational Autoencoder)

The Stage 1 architecture uses **Dual VAEs** to translate images from the "Old Photo" domain to the "Clean Photo" domain. During training, the VAE is evaluated on two distinct losses:

*   **Reconstruction Loss (L1 Loss):** 
    *   **What it is:** Measures how well the VAE can decode the latent space back into the original image. 
    *   **Score:** Decreases smoothly during training.
*   **KL Divergence (Kullback-Leibler Divergence):**
    *   **What it is:** Measures how closely the encoded latent space matches a standard normal distribution (a bell curve). 
    *   **Score:** The VAE balances this with Reconstruction loss to ensure the latent space is smooth and continuous, allowing for flawless domain translation (from old -> new).

---

## 4. Training Scores vs Testing Scores

If the examiner asks about your Training and Testing scores, here is how you answer for this specific architecture:

> **Viva Voce / Q&A Answer Strategy**
> 
> *"Because this is a Generative Adversarial Network (GAN), we don't have a traditional 'Training Accuracy' graph that goes up to 99%. Instead, our training score is a **Minimax Loss** between a Generator and a Discriminator. During training, the Generator's loss fluctuates as it tries to fool the Discriminator, while the Discriminator's loss fluctuates as it tries to catch fakes. They reach an equilibrium rather than a perfect score.* 
> 
> *Our testing scores are evaluated using **FID (Fréchet Inception Distance)** for realism and **PSNR/SSIM** for pixel accuracy on a synthetically degraded test set. In our architecture, the addition of the non-local attention blocks for scratch removal significantly boosts our PSNR over baseline Pix2Pix models."*

---

### How to show this to your examiner:
You can open the `Evaluation.ipynb` file (located in the project folder) in Jupyter Notebook or VSCode. It contains a script that:
1. Loads a clean image.
2. Programmatically damages it with synthetic scratches and Gaussian noise.
3. Runs it through your pipeline.
4. Outputs the exact **MSE, PSNR, and SSIM** scores side-by-side with the images!
