# cs2420

This repository contains code and notebooks related to a project involving fine-tuning and testing CLIP and Stable Diffusion models for forensic sketch image generation.

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.7+
*   CUDA-enabled GPU (recommended for faster training)
*   `pip` package installer

### Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/fiidalgo/cs2420.git
    cd cs2420
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Overview

This project explores the application of Stable Diffusion and CLIP models to the task of generating images from forensic sketches.  The goal is to improve the quality and accuracy of generated images by fine-tuning these models.

### Models Used

*   **Stable Diffusion (Hugging Face `runwayml/stable-diffusion-v1-5`):**  The project utilizes the `runwayml/stable-diffusion-v1-5` Stable Diffusion model from Hugging Face. This model is a latent text-to-image diffusion model capable of generating photorealistic images given any text input.  It was used as the base model for fine-tuning and testing.

*   **CLIP (Contrastive Language-Image Pre-training):** CLIP is a neural network trained on a variety of (image, text) pairs. It can be used to understand the relationship between images and text. In this project, CLIP is used in conjunction with Stable Diffusion to guide the image generation process.  The specific CLIP model used is implicitly defined by the Stable Diffusion pipeline.

### Experiment Setup

The project investigates the performance of the following setups:

1.  **Base Stable Diffusion Model:**  The performance of the pre-trained `runwayml/stable-diffusion-v1-5` model is evaluated directly for generating images from forensic sketch descriptions.

2.  **CLIP -> Stable Diffusion Pipeline:**  A pipeline is constructed where CLIP is used to encode the text description of the forensic sketch, and this encoding is then used to guide the Stable Diffusion model in generating the image.

3.  **Fine-Tuned CLIP -> Stable Diffusion Pipeline (with LoRA):**  This is the core of the project.  Both the CLIP and Stable Diffusion models are fine-tuned using LoRA (Low-Rank Adaptation).  LoRA allows for efficient fine-tuning by only training a small number of parameters.

    *   **Ablation Study:** Before fine-tuning, an ablation study was conducted to determine the optimal layers within both CLIP and Stable Diffusion to apply LoRA.  The directories `ablation_study_both`, `ablation_study_cross_only`, and `ablation_study_self_only` contain the results and code related to this study.  The goal was to identify the layers that contribute most to performance when fine-tuned.

## Key Features

*   **CLIP Fine-tuning:**  `clip_fine_tune.py` provides scripts for fine-tuning the CLIP model.
*   **Stable Diffusion Fine-tuning:** `sd_fine_tune.py` allows fine-tuning of the Stable Diffusion model.
*   **Testing Scripts:** `clip_sd_test.py` and `sd_test.py` contain scripts for testing the performance of the fine-tuned models.
*   **Ablation Studies:** The repository includes directories (`ablation_study_both`, `ablation_study_cross_only`, `ablation_study_self_only`) for conducting ablation studies to analyze the impact of different components.
*   **Metrics Analysis:** `final_metrics.ipynb` is a Jupyter Notebook for analyzing and visualizing the final metrics of the experiments.

## Troubleshooting

*   **CUDA issues:** Ensure you have the correct CUDA drivers installed and that your environment is configured to use the GPU.
*   **Dependency conflicts:** Double-check that all dependencies are installed correctly and that there are no version conflicts.  Try upgrading `pip` if you encounter issues.
*   **Memory errors:**  Fine-tuning large models can be memory-intensive.  Try reducing the batch size or using gradient accumulation to reduce memory usage.
