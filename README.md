# Fusion Steering: Prompt-Specific Activation Control

This repository contains the code and data for the Fusion Steering project. The main notebook, `Fusion_Steering_Notebook.ipynb`, demonstrates our experiments and results. This README provides instructions for running the notebook, describes the required files, and details how to reproduce our results.

## Directory Structure and Required Files

As provided in the ZIP file, place all of the following files in the same directory as `Fusion_Steering_Notebook.ipynb`:

- `corrected_responses.csv`
- `incorrect.csv`
- `mistral_grader_results_baseline.csv`
- `mistral_grader_results_full.csv`
- `mistral_grader_results_segmented.csv`
- `per_prompt_optuna_summary_full.csv`
- `per_prompt_optuna_summary_segmented.csv`
- `reference_vectors.pt`
- `simpleqa_initial_outputs.csv`

## How to Run and Reproduce Results

**All training, evaluation, and result reproduction steps can be performed directly in the notebook.**

1. Open [`Fusion_Steering_Notebook.ipynb`](./Fusion_Steering_Notebook.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Ensure all required files (listed above) are uploaded to the Colab environment or present in the same directory.
3. Go to `Runtime > Change runtime type` and select **GPU** (a free T4 is sufficient).
4. Run all cells in order. **No external scripts or command-line steps are required.** 

**Long-running cells** (e.g., model evaluation or hyperparameter tuning) are marked in the notebook's Markdown sections for your convenience.

## Results Table and Reproducibility

To reproduce and compare all results, you must run all cells in the notebook sequentially. The **Final Evaluation Data Assembly** section depends on intermediate files that are created by running the earlier sections (Baseline, Full-Layer, and Segmented Steering Scoring). Skipping directly to this section will result in missing outputs.

If you wish to examine or rerun the evaluation for a specific method, use the following notebook sections:

| Experiment/Method         | Notebook Section Title                | Output File(s)                                   | Input File(s)                                                                 |
|--------------------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------------------------------------|
| Baseline Steering        | `### Baseline Scoring`                | `mistral_grader_results_baseline_with_score_and_overlap.csv` | `incorrect.csv`, `mistral_grader_results_baseline.csv`                         |
| Full-Layer Steering      | `### Full-Layer Steering Scoring`     | `mistral_grader_results_full_merged.csv`         | `per_prompt_optuna_summary_full.csv`, `mistral_grader_results_full.csv`        |
| Segmented Steering       | `### Segmented Steering Scoring`      | `mistral_grader_results_segmented_with_score.csv`| `per_prompt_optuna_summary_segmented.csv`, `mistral_grader_results_segmented.csv` |
| Final Results Assembly   | `# **Final Evaluation Data Assembly**`| (Merged DataFrame for comparison and figures)     | All above intermediate output files                                           |

**Instructions:**
- For full reproducibility and to generate all summary charts, run all cells in order from the start of the notebook.
- The **Final Evaluation Data Assembly** section requires outputs from previous steps; do not skip earlier sections.
- For method-specific outputs, navigate to the corresponding section in the notebook as shown above.
- No external scripts are needed; all results are generated within the notebook.

## Dependencies

All required Python packages are installed automatically in the **first cell of the notebook**. For reference or for local runs, the full set of dependencies is:

- `bitsandbytes`
- `datasets`
- `optuna`
- `matplotlib`
- `nltk`
- `mistralai`
- `transformers`
- `huggingface_hub`
- `numpy`
- `torch`
- `pandas`
- `seaborn`

**Install command for Colab/local runs:**
```python
pip install -U bitsandbytes datasets optuna matplotlib nltk mistralai transformers huggingface_hub numpy torch pandas seaborn
```

### Dependency Table
| Package            | Reason Needed (Import/Use)                    |
|--------------------|-----------------------------------------------|
| bitsandbytes       | Quantization/efficient LLMs                   |
| datasets           | `load_dataset`                                |
| optuna             | Hyperparameter tuning                         |
| matplotlib         | Plotting                                      |
| nltk               | Tokenization                                  |
| mistralai          | Mistral API                                   |
| transformers       | HuggingFace models/tokenizers                 |
| huggingface_hub    | Model downloads, notebook login               |
| numpy              | Numerical ops                                 |
| torch              | PyTorch models                                |
| pandas             | DataFrames                                    |
| seaborn            | Plotting                                      |

> **Note:**
> - Standard Python libraries (`os`, `json`, `re`, `string`, `time`) do not require installation.
> - The notebook will attempt to download the correct NLTK tokenizer with `nltk.download('punkt')`. If running locally, ensure you have internet access for this step.


## Notes
- For reviewer convenience, all long-running cells are clearly marked in the notebook.
- If you encounter issues with missing dependencies or files, please check that all files listed above are present and that the Colab environment is using a GPU.
