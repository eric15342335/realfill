# 🖼️ RealFill: Reproduction, Exploration, and Improvement

**APAI3010/STAT3010 Image Processing and Computer Vision - Group Project (Spring 2025)**
The University of Hong Kong

This repository contains the code and resources for our group project focused on the paper **"RealFill: Reference-Driven Generation for Authentic Image Completion"** by Tang et al. (SIGGRAPH 2024). Our objective was to reproduce the core results, analyze the method's strengths and weaknesses, and explore potential extensions.

---

## 🚀 Quick Links

* **[📜 Project Report (PDF)](https://github.com/eric15342335/realfill/releases/download/project/APAI3010_Project_Report_compressed_linearize.pdf)**: Detailed analysis, findings, and discussion.
* **[📊 Presentation Slides (PPTX)](https://github.com/eric15342335/realfill/releases/download/project/APAI3010.RealFill.Group.3.pptx)**: Summary presentation of our work.
* **[📹 Presentation Video (MP4)](https://github.com/eric15342335/realfill/releases/download/project/APAI3010.RealFill.Group.3.mp4)**: Recording of the presentation slideshow.
* **[📄 Original RealFill Paper (PDF)](https://doi.org/10.1145/3658237)**: The paper our project is based on.
* **[📝 Report LaTeX Template (NeurIPS 2024)](https://github.com/eric15342335/realfill/releases/download/project/neurips_2024.pdf)**: The template used for the project report.
* **[📄 FaithFill Paper (PDF)](https://arxiv.org/abs/2406.07865)**: Relevant paper for extension inspiration.
* **[📄 LoFTR Paper (PDF)](https://arxiv.org/abs/2104.00680)**: Relevant paper for LoFTR feature matcher.
* **[💻 Google Colab Notebook](./train_realfill.ipynb)**: The primary environment used for experiments. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./train_realfill.ipynb) *(Note: Requires setting up data and potentially Drive access within Colab)*

---

## 🎯 Project Overview

Image completion, particularly achieving *authentic* results faithful to the original scene, is a challenging task. RealFill tackles this by fine-tuning a diffusion inpainting model (Stable Diffusion v2 Inpainting) using a small set of reference images and Low-Rank Adaptation (LoRA).

This project involved:

1. **Reproduction:** Implementing the RealFill pipeline and reproducing key results from the paper on the RealBench dataset.
2. **Exploration & Analysis:** Evaluating the reproduced model's performance, identifying limitations (especially concerning geometric consistency and computational cost), and testing on custom real-world data.
3. **Extension (ReFill):** Proposing and implementing a 2-stage iterative refinement process ("ReFill") using LoFTR-ranked generated images as augmented references, inspired by related works like FaithFill.
4. **Benchmarking:** Developing a comprehensive benchmarking suite to evaluate image completion quality using various metrics (PSNR, SSIM, LPIPS, DreamSim, DINO, CLIP).

## 🧑‍💻 Team Members

<!-- markdownlint-disable MD033 -->

<table>
    <tbody>
        <tr>
            <th>Name</th>
            <th>UID</th>
            <th>Profile</th>
        </tr>
        <tr>
            <td>Cheng Ho Ming</td>
            <td>3036216734</td>
            <td><a href="https://github.com/eric15342335"><img src="https://avatars.githubusercontent.com/u/70310617" alt="Cheng Ho Ming" width=50></a></td>
        </tr>
        <tr>
            <td>Chung Shing Hei</td>
            <td>3036216760</td>
            <td><a href="https://github.com/MaxChungsh"><img src="https://avatars.githubusercontent.com/u/70740754" alt="Chung Shing Hei" width=50></a></td>
        </tr>
        <tr>
            <td>Chan Hin Chun</td>
            <td>3036218017</td>
            <td><a href="https://github.com/JChan-cs"><img src="https://avatars.githubusercontent.com/u/158464686" alt="Chan Hin Chun" width=50></a></td>
        </tr>
    </tbody>
</table>

<!-- markdownlint-enable MD033-->

*(See Appendix A.1 in the [Project Report](https://github.com/eric15342335/realfill/releases/download/project/APAI3010_Project_Report_compressed_linearize.pdf) for a detailed breakdown of contributions.)*

## 📁 Repository Structure

```sh
├── benchmark/                     # Scripts for individual metric calculations (PSNR, SSIM, LPIPS, etc.)
├── data/                          # Placeholder for example data (full datasets usually downloaded separately)
├── project_documents/             # Contains the final report LaTeX template
├── README-Realfill.md             # Original README from the forked base repository
├── LICENSE                        # MIT License file covering base code and our modifications
├── benchmarks.py                  # Main script to orchestrate metric calculation and analysis
├── infer.py                       # Script for running inference with a trained RealFill model
├── loftr_ranking.py               # Script for ranking images based on LoFTR correspondences
├── requirements.txt               # Core dependencies for training and inference
├── requirements-benchmarks.txt    # Additional dependencies for the benchmarking suite
├── train_realfill.ipynb           # Jupyter Notebook for running experiments (primarily on Google Colab)
├── train_realfill.py              # Python script for training/fine-tuning the RealFill model
```

## ⚙️ Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/eric15342335/realfill
    cd realfill
    ```

2. **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```

3. **Install Dependencies:**
    * For **Training & Inference**:

        ```bash
        # Using pip:
        pip install -r requirements.txt
        # Or using the faster uv:
        # uv pip install -r requirements.txt
        ```

    * For **Benchmarking**:

        ```bash
        # Using pip:
        pip install -r requirements-benchmarks.txt
        # Or using uv:
        # uv pip install -r requirements-benchmarks.txt
        ```

    > ⚠️ **GPU Acceleration (PyTorch):**
    > The `requirements.txt` file installs the **CPU-only** version of PyTorch by default to ensure basic compatibility. For GPU acceleration (highly recommended for training and faster inference/benchmarking), you **must manually install the appropriate GPU-enabled version** of PyTorch matching your CUDA version *after* installing requirements.
    > Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions.

4. **Dataset:**
    * The experiments primarily use the `RealBench` dataset subset provided by the original RealFill authors and our custom dataset.
    * The `train_realfill.ipynb` notebook includes cells to download and extract the necessary datasets (`realfill_data_release_full.zip`, `jensen_images.zip`) within the Colab environment. Follow instructions there.
    * For local use, download these datasets manually and place them in the appropriate directory structure (e.g., `./realfill_data_release_full/`, `./jensen_images/`).

## ▶️ Usage

The primary workflow for this project was developed and tested using the **[`train_realfill.ipynb`](./train_realfill.ipynb)** notebook, especially on Google Colab. We recommend using it for reproducing experiments.

Alternatively, you can use the Python scripts directly as follows:

### 1. Training / Fine-tuning

* Use the `train_realfill.py` script launched via `accelerate`.
* **Basic Configuration:** Set environment variables or command-line arguments for:
  * `--pretrained_model_name_or_path`: Base model (e.g., `stabilityai/stable-diffusion-2-inpainting`).
  * `--train_data_dir`: Path to the specific scene directory.
  * `--output_dir`: Where to save LoRA checkpoints/model.
  * Other core parameters (learning rate, steps, LoRA rank/alpha).

#### Running on Low-Memory GPUs (e.g., Google Colab T4 - 16GB VRAM) 🔋

Training RealFill typically requires significant VRAM. To successfully run fine-tuning on hardware with limited memory like the 16GB GPUs available on Google Colab Free Tier, several optimizations are **essential**:

1. **Mixed Precision:** Enable FP16 mixed precision via Accelerate configuration or by passing `--mixed_precision=fp16` (if overriding config).
2. **8-bit Adam Optimizer:** Use the memory-efficient Adam variant via `--use_8bit_adam`. Requires `bitsandbytes`.
3. **xFormers:** Enable memory-efficient attention mechanisms with `--enable_xformers_memory_efficient_attention`. Requires `xformers`.
4. **Set Grads to None:** Further reduce memory by setting gradients to `None` instead of zeroing them using `--set_grads_to_none`.

* **Example Command (Adapted from our Colab Setup for RealBench Scene 23):**
    This command incorporates the necessary flags for low-memory training and includes monitoring/checkpointing flags (see next section).

    ```bash
    # --- Set Environment Variables ---
    export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
    export BENCHMARK="RealBench"
    export DATASET_NUMBER=23
    export TRAIN_DIR="realfill_data_release_full/$BENCHMARK/$DATASET_NUMBER"
    export OUTPUT_DIR="$BENCHMARK-$DATASET_NUMBER-model" # Example output dir

    # --- Launch Training ---
    accelerate launch train_realfill.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --train_data_dir=$TRAIN_DIR \
      --output_dir=$OUTPUT_DIR \
      --resolution=512 \
      --train_batch_size=16 \
      --gradient_accumulation_steps=1 \
      --use_8bit_adam `# Use 8-bit Adam` \
      --enable_xformers_memory_efficient_attention `# Use xFormers` \
      --set_grads_to_none `# Set Grads to None` \
      --unet_learning_rate=2e-4 \
      --text_encoder_learning_rate=4e-5 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=100 \
      --max_train_steps=2000 \
      --lora_rank=8 \
      --lora_dropout=0.1 \
      --lora_alpha=16 \
      --mixed_precision=fp16 `# Explicitly set mixed precision` \
      --resume_from_checkpoint="latest" `# Resume if checkpoints exist` \
      --report_to tensorboard `# Enable TensorBoard logging` \
      --checkpointing_steps 200 `# Save checkpoint every 200 steps` \
      --validation_steps 100 `# Run validation every 100 steps` \
      --num_validation_images 4 `# Generate 4 validation images`
    ```

    Some options worth considering:

  * `--gradient_checkpointing`: Enable gradient checkpointing to *save memory* at the *cost of speed*. This is useful if you trying to run on a GPU with less VRAM than Google Colab T4 (16GB).
  * `--allow_tf32`: Enable TensorFloat-32 (TF32) for NVIDIA Ampere GPUs (e.g., A100, RTX 30/40 series). This can improve performance but is not available on all GPUs.
  * `--mixed_precision=bf16`: Use Brain Float 16 (BF16) precision if supported by your hardware. This is generally faster than FP16 but requires specific GPU support (e.g., A100, H100). Google Colab T4 does not support BF16.

    Note that by default `--train_batch_size` has no effect if the number is larger than the number of available images in the training set (reference & target images). If you have access to hardware with more VRAM, you can consider using `--pad_to_full_batch` to pad the input batch to the full batch size.

#### Monitoring Training with TensorBoard 📊

Monitoring the training process is crucial, especially to see how well the model is learning to inpaint the target region during fine-tuning. We incorporated TensorBoard logging for this purpose.

* **Enable Logging:** Add the `--report_to tensorboard` flag to your `accelerate launch` command.
* **Configure Validation:** Use the following flags to control how often validation occurs and how many images are generated:
  * `--validation_steps <N>`: Runs the validation loop every `N` training steps. Validation involves generating sample inpainted images using the current state of the model.
  * `--num_validation_images <K>`: Generates `K` sample images during each validation run.
* **Checkpointing:** Validation is often tied to checkpointing. The `--checkpointing_steps <M>` flag saves the model state every `M` steps.
* **Viewing Logs:** While training is running (or after it finishes), navigate to the parent directory of your `OUTPUT_DIR` in your terminal and run:

    ```bash
    tensorboard --logdir <OUTPUT_DIR>/logs
    ```

    (Note: `<OUTPUT_DIR>` is the directory specified in your training command, e.g., `RealBench23-model`).
    Open the URL provided by TensorBoard (usually `http://localhost:6006/`) in your browser. The generated validation images will appear under the "Images" tab, allowing you to visually inspect the learning progress.

### 2. Inference

* Use the `infer.py` script after training.
* **Configuration:**
  * `--model_path`: Path to the trained model directory (containing `unet/` and `text_encoder/` subfolders).
  * `--validation_image`: Path to the target image (`target.png`).
  * `--validation_mask`: Path to the mask image (`mask.png`).
  * `--output_dir`: Directory to save the 16 generated output images.
* **Example Command:**

    ```bash
    accelerate launch infer.py \
      --model_path="./RealBench23-model" \
      --validation_image="./realfill_data_release_full/RealBench/23/target/target.png" \
      --validation_mask="./realfill_data_release_full/RealBench/23/target/mask.png" \
      --output_dir="./realfill_results/RealBench23-results"
    ```

### 3. Benchmarking 📈

* Use the `benchmarks.py` script to evaluate generated results against ground truth.
* **Configuration:**
  * `--results_base_dir`: The parent directory containing multiple scene result folders (e.g., `./realfill_results/` or your Google Drive path). Folders should follow a pattern like `RealBench-X-results` or `Custom-Y-results`.
  * `--realbench_dataset_dir`: Path to the base directory of the *original* RealBench dataset (needed for finding corresponding GT/Mask files).
  * `--custom_dataset_dir`: Path to the base directory of the custom dataset (if applicable).
  * `--cache_dir`: Directory to store intermediate and final metric caches (speeds up re-runs).
  * `--output_file`: Path to save the final text report.
  * `--metrics`: (Optional) List specific metrics to run (e.g., `PSNR SSIM LPIPS`). Default is all configured metrics.
  * `--force_recalc`: (Optional) Force recalculation for specific metrics or `all` or `loftr`.
* **Functionality:**
  * Discovers result folders matching the pattern.
  * Finds corresponding ground truth (`gt.png`) and mask (`mask.png`) in the dataset directories.
  * Calls individual metric scripts in `benchmark/` in parallel.
  * Uses caching heavily (`master_results_cache.json` and `per_scene_cache/`) to avoid recomputing.
  * Runs LoFTR ranking (`loftr_ranking.py --rank-only`) on RealBench results if the script is found.
  * Performs comparative analysis (FP16/32, Gen/NonGen, LoFTR filtering).
  * Generates and saves a formatted report (`benchmark_report.txt`).
* **Example Command:**

    ```bash
    python benchmarks.py \
      --results_base_dir="./realfill_results" \
      --realbench_dataset_dir="./realfill_data_release_full" \
      --custom_dataset_dir="./jensen_images" \
      --cache_dir="./benchmark_cache" \
      --output_file="./benchmark_report.txt"
    ```

### 4. LoFTR Ranking/Selection

* The `loftr_ranking.py` script serves two purposes:
    1. **Reference Selection (within `train_realfill.ipynb`):** When `USE_GENERATED_REF_IMAGE=True` in the notebook, this script is called to rank images from a *previous* run's output directory against the original references. It copies the top N candidates (based on `--target-count`) into the *current* run's `ref/` directory before training starts.
    2. **Result Ranking (by `benchmarks.py`):** The benchmarking script calls `loftr_ranking.py --rank-only`. This ranks the 16 generated images within a result folder against the original references and saves the ranking scores to `loftr_ranking_scores.json` inside that result folder. This data is used for the LoFTR filtering analysis in the final report.

## ✨ Our Extension: ReFill

Based on our analysis and inspiration from concurrent work [FaithFill](https://github.com/eric15342335/realfill/releases/download/project/FaithFill.pdf), we proposed *ReFill*. The core idea is a two-stage iterative refinement:

1. Run the standard RealFill fine-tuning and inference process.
2. Use `loftr_ranking.py` to identify the best-generated images from step 1 based on correspondence with the original references.
3. Augment the original reference set with these top-ranked generated images (up to a limit, e.g., 5 total references).
4. Perform a *second* RealFill fine-tuning pass using this augmented reference set.
5. Run inference again using the model from step 4.

The hypothesis was that adding high-quality, view-diverse generated references could improve the model's understanding of the scene geometry and lead to more authentic completions.

*(See Section 4 in the [Project Report](https://github.com/eric15342335/realfill/releases/download/project/APAI3010_Project_Report_compressed_linearize.pdf) for implementation details and results.)*

## 📊 Results

Detailed quantitative results, qualitative examples, comparisons, and analysis of both the baseline RealFill reproduction and our ReFill extension can be found in the **[Project Report](https://github.com/eric15342335/realfill/releases/download/project/APAI3010_Project_Report_compressed_linearize.pdf)**.

Key findings include:

* Successful reproduction of RealFill, achieving comparable (and sometimes better on specific metrics like LPIPS) performance to the original paper using FP16 mixed-precision.
* Confirmation of RealFill's strengths (authenticity, handling variations) and weaknesses (geometric inconsistency, dependence on reference quality, computational cost).
* Our ReFill extension showed only marginal quantitative improvements and subtle qualitative gains, suggesting that LoFTR-based selection of 2D views might not be sufficient to significantly enhance 3D-aware completion without more explicit geometric priors.

## 📄 License

This repository builds upon the unofficial RealFill implementation by [thuanz123](https://github.com/thuanz123/realfill), which is licensed under MIT.

Our project, including all modifications, extensions (*ReFill*), benchmarking suite, and custom code, is also released under the MIT License. This permits anyone to use, modify, and distribute this software, provided the original copyright notice and permission notice are included.

See the [LICENSE](./LICENSE) file for full details.

## 🙏 Acknowledgements

* The original **RealFill authors** (Tang et al.) for their foundational work and the release of the RealBench dataset, which enabled our reproduction and analysis.
* **thuanz123** for the unofficial implementation which served as our starting point.
* The creators of the numerous open-source libraries used, including **PyTorch, Diffusers, Transformers, PEFT, bitsandbytes, xFormers, Kornia, Accelerate, LoFTR, LPIPS, DreamSim**, and others.
* **Prof. Kai Han** and **Mr. Weining Ren** (TA) at HKU for their invaluable guidance and feedback throughout the project.
* **Google Colab** for providing the necessary computational resources.

---

[Back to top](#️-realfill-reproduction-exploration-and-improvement)
