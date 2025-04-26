# What is this repo?

This repo serves as the backup of the [Realfill unofficial implementation](https://github.com/thuanz123/realfill), with our own modifications in order to run on Google Colab and on our own devices. This repo is associated with [APAI3010 Image Processing and Computer Vision](https://webapp.science.hku.hk/sr4/servlet/enquiry?Type=Course&course_code=APAI3010).

## Team members

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

## Setup

1. **Prerequisites:**
    * Google Account (for Colab and Drive)
    * Basic familiarity with Python and Jupyter Notebooks.

2. **Clone Repository:**

    ```bash
    git clone https://github.com/eric15342335/realfill
    cd realfill
    ```

3. **Install Dependencies:**
    * For **Training and Inference**, use the primary requirements file:

        ```bash
        # Ideally within a virtual environment
        pip install -r requirements.txt
        # or you can use `uv` for faster installation
        # uv pip install -r requirements.txt
        ```

    * For **Running the Benchmarking Suite**, install additional/specific dependencies:

        ```bash
        # Ideally within the same virtual environment
        pip install -r requirements-benchmarks.txt
        # uv pip install -r requirements-benchmarks.txt
        ```

    * *(Note: The `train_realfill.ipynb` notebook handles dependency installation within the Colab environment).*

4. **Dataset:**
    * The necessary datasets (like `realfill_data_release_full.zip` and `jensen_images.zip` mentioned in the notebook) need to be downloaded.
    * The `train_realfill.ipynb` notebook contains cells to download and extract these datasets into the Colab environment. Follow the instructions within the notebook.

## Usage (Google Colab Workflow)

The primary way to use this repository is through the `train_realfill.ipynb` notebook on Google Colab.

1. **Open in Colab:** Upload the notebook to your Google Drive and open it with Colab, or use the "Open In Colab" badge if configured.
2. **Configure Runtime:** Ensure the runtime type is set to `GPU` (e.g., T4 or V100).
3. **Run Setup Cells:** Execute the initial cells in the notebook to:
    * Clone the repository (if not already running from a cloned environment).
    * Install necessary dependencies (`requirements.txt`).
    * Download and unzip the datasets.
4. **Configure Experiment:** Modify the configuration cell (around cell index 7 or 8):
    * `USE_DRIVE_STORAGE`: Set to `True` (recommended) to save models, results, and cache to your Google Drive. Mounts Drive when `True`.
    * `USE_FP32`: Set to `False` for default mixed-precision (FP16) or `True` for FP32 training. Affects output folder names.
    * `USE_GENERATED_REF_IMAGE`: Set to `True` to enable the LoFTR selection step, where results from a previous run are ranked and used as additional references for the current run. Requires a corresponding non-generated results folder to exist.
    * `TARGET_TOTAL_REF_COUNT`: Target number of reference images when `USE_GENERATED_REF_IMAGE` is `True`.
    * `DATASET_NUMBER`: **Crucially, set this** to the specific scene number (e.g., `"22"`, `"15"`, `"35"`) you want to process from the `RealBench` or `Custom` dataset splits.
    * Paths (`DRIVE_BASE_DIR`, `OUTPUT_DIR`, `OUTPUT_IMG_DIR`, `REF_DIR`, `SOURCE_RESULTS_DIR_FOR_COPY`, etc.) are constructed automatically based on these settings. Review the printed paths to ensure they are correct.
5. **LoFTR Reference Selection (Optional):** If `USE_GENERATED_REF_IMAGE` is `True`, the configuration cell will execute `loftr_ranking.py` to select and copy the best candidates from the specified source results directory into the current run's reference (`ref`) directory.
6. **Train the Model:** Run the `accelerate launch train_realfill.py ...` cell to fine-tune the RealFill model for the selected scene. Checkpoints and the final LoRA weights will be saved to the `OUTPUT_DIR` (likely on Google Drive).
7. **Run Inference:** Execute the `accelerate launch infer.py ...` cell to generate the 16 output images using the trained model. Results are saved to `OUTPUT_IMG_DIR` (likely on Google Drive).
8. **Run Benchmarking:**
    * Execute the cell that installs benchmark requirements (`pip install -r requirements-benchmarks.txt`).
    * Execute the cell that runs `benchmarks.py`. This script will:
        * Automatically discover all relevant result folders in your `DRIVE_BASE_DIR`.
        * Run all configured metrics (or all 6 by default) for each result set.
        * Utilize the cache located in `<DRIVE_BASE_DIR>/benchmark_cache` to speed up subsequent runs.
        * Perform the comparative analysis (FP16/32, Gen/NonGen, LoFTR filtering).
        * Print a formatted report to the cell output.
        * Save the same report to `<DRIVE_BASE_DIR>/benchmark_report.txt`.

## Benchmarking System Details

* **Scripts:** Located in `benchmark/`. Each script calculates one metric for a single scene and manages its own per-scene cache.
* **Orchestrator:** `benchmarks.py` drives the process, calls individual scripts, aggregates results, performs analysis, and generates the report.
* **Caching:**
  * Per-scene caches (including per-image scores) are stored in `<cache_dir>/per_scene_cache/<metric_name>/<folder_name>.json`.
  * A master cache of average scores is stored in `<cache_dir>/master_results_cache.json`.
  * LoFTR ranking results (from `--rank-only` mode) are stored in `<results_folder>/loftr_ranking_scores.json`.
* **Output:** A detailed text report (`benchmark_report.txt` by default) is generated, containing tables comparing different configurations and metrics.

## LoFTR Reference Selection / Ranking

The `loftr_ranking.py` script compares candidate images (e.g., from a previous inference run) against a set of reference images using LoFTR feature matching.

* **Selection Mode (Default):** Used within the notebook (`USE_GENERATED_REF_IMAGE=True`). Ranks candidates and copies the top N (based on `--target-count`) to the reference directory for the *next* training run.
* **Rank-Only Mode (`--rank-only`):** Used by `benchmarks.py`. Ranks candidates and saves the scores/filenames to a JSON file (default: `loftr_ranking_scores.json`) within the candidate directory. This ranking is used for the filtering analysis.

## License

We have not yet decided on the license for this repository. The original RealFill code is licensed under the [MIT License](./LICENSE).

For the another group doing Realfill: please do not copy our code without our explicit written permission. Thanks! -- @eric15342335

## Acknowledgements

* The original RealFill authors.
* The creators of the diffusion models, LoFTR, and benchmark libraries used.
* Our course professor and teaching assistant for their guidance and support.
* So many people to thank with, we are unable to list them all here. We are grateful for the open-source community and the resources available online.

[Back to top](#what-is-this-repo)
