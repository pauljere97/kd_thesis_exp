# Knowledge Distillation Experiments for LLMs

This repository contains the experimental code for **Chapter 4** of the thesis on Knowledge Distillation of Large Language Models.

## Overview

The experiments evaluate various knowledge distillation techniques to compress large language models while preserving performance.

### Models
- **Teacher**: Mistral-7B-Instruct (7B parameters)
- **Student**: TinyLlama-1.1B-Chat + quantized variants

### Tasks
- **SST-2**: Sentiment classification (GLUE benchmark)
- **SQuAD v1.1**: Extractive question answering

### Methods
| Method | Description |
|--------|-------------|
| Baseline (B0) | Fine-tuned student without distillation |
| Logit KD (KD1) | Soft label matching with temperature scaling |
| Feature KD (KD3) | Hidden state alignment between teacher/student |
| Sequence KD (KD2) | Sequence-level distillation for QA |
| Quantization | 8-bit and 4-bit quantized student models |

## Project Structure

```
├── chapter4_experiments.ipynb   # Main experiment notebook
├── results/
│   ├── figures/                 # Generated visualizations
│   │   ├── fig1_quality_vs_size.png
│   │   ├── fig2_latency_vs_size.png
│   │   ├── fig3_throughput.png
│   │   ├── fig4_memory.png
│   │   └── fig5_pareto.png
│   └── summary/                 # Aggregated results
│       ├── main_results.csv
│       ├── ablation_results.csv
│       └── efficiency_results.csv
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Note**: A GPU with at least 16GB VRAM is recommended. The notebook is designed for Google Colab with GPU runtime.

## Usage

1. Open `chapter4_experiments.ipynb` in Jupyter or Google Colab
2. Configure phase switches in Section 1 to control which experiments to run:
   ```python
   RUN_TEACHER_CACHE = True   # Cache teacher outputs
   RUN_B0 = True              # Baseline training
   RUN_KD1 = True             # Logit-based KD
   RUN_KD3 = True             # Feature-based KD
   RUN_KD2 = True             # Sequence-level KD
   RUN_QUANT = True           # Quantization experiments
   RUN_BENCHMARK = True       # Efficiency benchmarks
   RUN_PLOTS = True           # Generate figures
   ```
3. Run all cells sequentially

## Key Hyperparameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| Temperature (τ) | 2.0, 4.0 | KD softmax temperature |
| Alpha (α) | 0.5, 0.7, 0.9 | CE vs KL loss weight |
| Lambda (λ) | 0.1, 0.5, 1.0 | Feature MSE weight |
| LoRA rank | 16 | Low-rank adaptation dimension |
| Quantization | 8-bit, 4-bit | Student model precision |

## Results

Results are saved to `results/summary/` as CSV files and visualizations to `results/figures/`.

### Metrics
- **Accuracy**: Classification accuracy (SST-2)
- **F1**: Exact match / F1 score (SQuAD)
- **Latency**: Inference time (ms)
- **Throughput**: Tokens per second
- **Memory**: GPU memory usage (MB)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- ~16GB GPU VRAM (for teacher model inference)

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{kd_llm_thesis,
  title={Knowledge Distillation of Large Language Models},
  author={},
  year={2026}
}
```

## License

This project is for academic research purposes.
