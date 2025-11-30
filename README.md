# SAM3 Fine-Tuning and Prediction Framework: SAM3 vs YOLO11 for Agricultural Instance Segmentation

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Composite](https://img.shields.io/badge/License-Composite-yellow.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Applied--AI--Research--Lab-blue)](https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation)

**Citation:**
```bibtex
@software{sam3_roumeliotis_sapkota_2025,
  author = {Roumeliotis, Konstantinos I. and Sapkota, Ranjan},
  title = {SAM3 Fine-Tuning Framework for Agricultural Instance Segmentation},
  year = {2025},
  institution = {University of the Peloponnese, Department of Informatics and Telecommunications, Greece and Cornell University, Biological & Environmental Engineering, Ithaca, New York, USA},
  url = {https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation}
}
```

> **Comprehensive framework for SAM3 model fine-tuning, SAM3 zero-shot predictions, SAM3 LoRA adaptation, and direct comparison with YOLO11 on agricultural datasets**

## 🌟 Overview

This project provides a **production-ready framework for SAM3 (Segment Anything Model 3) fine-tuning and prediction** on agricultural computer vision tasks. Compare **SAM3 vs YOLO11** performance across multiple training strategies including:

- ✅ **SAM3 Zero-Shot Predictions** - Baseline performance without training
- ✅ **SAM3 Full Fine-Tuning** - Complete model adaptation for agriculture
- ✅ **SAM3 LoRA Fine-Tuning** - Parameter-efficient training for SAM3
- ✅ **YOLO11 Instance Segmentation** - State-of-the-art comparison
- ✅ **Agricultural Datasets** - MinneApple & WeedsGalore pre-configured
- ✅ **Automated Evaluation** - Comprehensive metrics (IoU, F1, precision, recall)
- ✅ **Visual Comparison Reports** - Side-by-side prediction analysis

### 🎯 Perfect for:
- **SAM3 agriculture applications** (crop monitoring, yield estimation, weed detection)
- **SAM3 model fine-tuning research** and experimentation
- **SAM3 vs YOLO11 benchmarking** on custom datasets
- **SAM3 prompt engineering** and optimization
- **Parameter-efficient SAM3 training** with LoRA

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.7+ required
python --version

# GPU with CUDA support recommended (optional but 10x faster)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation.git
cd SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

```bash
# Download both agricultural datasets (MinneApple + WeedsGalore)
./datasets.sh both

# Or download individually
./datasets.sh minneapple
./datasets.sh weedsgalore

# Verify dataset integrity
./datasets.sh check
```

## 📖 Complete Workflow Guide

### 1️⃣ SAM3 Zero-Shot Predictions (No Training Required)

Run **SAM3 zero-shot segmentation** to establish baseline performance:

```bash
# SAM3 zero-shot on MinneApple dataset
./prediction_sam3_zero.sh minneapple

# SAM3 zero-shot on WeedsGalore dataset
./prediction_sam3_zero.sh weedsgalore

# Run on both datasets
./prediction_sam3_zero.sh both
```

**Results saved to:** `results/sam3_{dataset}_{timestamp}/`

### 2️⃣ Train Models (Fine-Tuning)

#### SAM3 Full Fine-Tuning

Train **SAM3 with full parameter updates** for maximum performance:

```bash
# Fine-tune SAM3 on MinneApple
./ft_sam3_full.sh minneapple

# Fine-tune SAM3 on WeedsGalore
./ft_sam3_full.sh weedsgalore

# Train on both datasets sequentially
./ft_sam3_full.sh both
```

**Models saved to:** `ft/sam3/full/{dataset}/best.pt`

#### SAM3 LoRA Fine-Tuning (Parameter-Efficient)

Train **SAM3 with LoRA adapters** (100x fewer parameters):

```bash
# SAM3 LoRA fine-tuning on MinneApple
./ft_sam3_lora.sh minneapple

# SAM3 LoRA fine-tuning on WeedsGalore
./ft_sam3_lora.sh weedsgalore

# Train on both datasets
./ft_sam3_lora.sh both
```

**Models saved to:** `ft/sam3/lora/{dataset}/best_lora/`

#### YOLO11 Training

Train **YOLO11 instance segmentation** for comparison:

```bash
# Train YOLO11 on MinneApple
./ft_yolo11.sh minneapple

# Train YOLO11 on WeedsGalore
./ft_yolo11.sh weedsgalore

# Train on both datasets
./ft_yolo11.sh both
```

**Models saved to:** `ft/yolo11/{dataset}/best.pt`

### 3️⃣ Generate Predictions with Fine-Tuned Models

#### SAM3 Full Fine-Tuned Predictions

```bash
# Predict with fine-tuned SAM3 model
./prediction_sam3_full.sh minneapple
./prediction_sam3_full.sh weedsgalore
./prediction_sam3_full.sh both
```

**Results saved to:** `results/sam3_full_{dataset}_{timestamp}/`

#### SAM3 LoRA Predictions

```bash
# Predict with SAM3 LoRA-adapted model
./prediction_sam3_lora.sh minneapple
./prediction_sam3_lora.sh weedsgalore
./prediction_sam3_lora.sh both
```

**Results saved to:** `results/sam3_lora_{dataset}_{timestamp}/`

#### YOLO11 Predictions

```bash
# Predict with trained YOLO11 model
./prediction_yolo11.sh minneapple
./prediction_yolo11.sh weedsgalore
./prediction_yolo11.sh both
```

**Results saved to:** `results/yolo11_{dataset}_{timestamp}/`

### 4️⃣ Complete Comparison Workflow

Run the **complete SAM3 vs YOLO11 comparison pipeline**:

```bash
# Step 1: Download datasets
./datasets.sh both

# Step 2: Train all models (can run in parallel on multiple GPUs)
./ft_sam3_full.sh both
./ft_sam3_lora.sh both
./ft_yolo11.sh both

# Step 3: Generate all predictions
./prediction_sam3_zero.sh both    # Baseline (no training)
./prediction_sam3_full.sh both    # Full fine-tuning
./prediction_sam3_lora.sh both    # LoRA fine-tuning
./prediction_yolo11.sh both       # YOLO11

# Step 4: Compare results in respective directories
ls -lh results/
```

## 📊 What This Project Does

### 🔬 Research Capabilities

This framework enables comprehensive **SAM3 model evaluation and fine-tuning** for agricultural computer vision:

1. **SAM3 Zero-Shot Baseline**: Evaluate SAM3's foundation model capabilities without domain-specific training
2. **SAM3 Fine-Tuning Comparison**: Compare full fine-tuning vs LoRA parameter-efficient training
3. **SAM3 vs YOLO11 Benchmarking**: Direct performance comparison on identical test sets
4. **Agricultural Instance Segmentation**: Apple detection/counting and weed identification
5. **Automated Evaluation Pipeline**: IoU, precision, recall, F1-score, counting accuracy
6. **Visual Analysis**: Side-by-side prediction comparisons and error analysis

### 📈 Evaluation Metrics

All predictions are automatically evaluated with:

- **Instance IoU** (Intersection over Union)
- **Pixel-level Precision & Recall**
- **F1 Score** (harmonic mean of precision/recall)
- **Counting Accuracy** (MAE, relative error)
- **Boundary Precision** (edge-specific metrics)
- **Per-image Performance Breakdown**

### 📁 Results Structure

Each prediction run generates:

```
results/sam3_full_minneapple_20251130_143022/
├── masks/                          # Raw predicted segmentation masks
├── visualizations/                 # Overlay images & comparisons
│   ├── overlays/                  # Predictions on original images
│   ├── comparisons/               # Side-by-side ground truth vs prediction
│   └── difference_maps/           # Error visualization
├── metrics/
│   ├── evaluation_metrics.csv     # Per-image detailed metrics
│   ├── evaluation_metrics.json    # Aggregate statistics
│   └── summary.txt               # Human-readable summary
└── report/
    ├── report.html               # Interactive web report
    └── evaluation_summary.json   # Complete evaluation data
```

## 🏗️ Project Architecture

### Directory Structure

```
Sam3/
├── datasets.sh                    # Centralized dataset management
├── ft_sam3_full.sh               # SAM3 full fine-tuning script
├── ft_sam3_lora.sh               # SAM3 LoRA fine-tuning script
├── ft_yolo11.sh                  # YOLO11 training script
├── prediction_sam3_zero.sh       # SAM3 zero-shot predictions
├── prediction_sam3_full.sh       # SAM3 full fine-tuned predictions
├── prediction_sam3_lora.sh       # SAM3 LoRA predictions
├── prediction_yolo11.sh          # YOLO11 predictions
├── requirements.txt              # Python dependencies
├── data/                         # Datasets (auto-created)
│   ├── minneapple/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── weedsgalore/
│       ├── train/
│       ├── val/
│       └── test/
├── ft/                           # Fine-tuned models (auto-created)
│   ├── sam3/
│   │   ├── full/{dataset}/      # Full fine-tuned SAM3 checkpoints
│   │   └── lora/{dataset}/      # LoRA adapter checkpoints
│   └── yolo11/{dataset}/        # YOLO11 checkpoints
├── results/                      # Prediction results (auto-created)
│   └── {model}_{dataset}_{timestamp}/
├── src/                          # Core utilities
│   ├── download_dataset.py      # Dataset acquisition
│   ├── evaluation.py            # Metrics computation
│   ├── visualization.py         # Result visualization
│   ├── data_loader.py           # Data loading utilities
│   ├── segmentation.py          # Segmentation helpers
│   ├── batch_processing.py      # Batch inference
│   ├── advanced_prompting.py    # SAM3 prompt strategies
│   └── latex_report.py          # Report generation
└── models/                       # Model-specific code
    ├── sam3/
    │   ├── predict_sam3_zero.py      # Zero-shot inference
    │   ├── convert_to_sam3_format.py # Data preprocessing
    │   ├── full/
    │   │   ├── train_sam3_full.py    # Full fine-tuning
    │   │   └── predict_sam3_full.py  # Full FT prediction
    │   └── lora/
    │       ├── train_sam3_lora.py    # LoRA fine-tuning
    │       └── predict_sam3_lora.py  # LoRA prediction
    └── yolo11/
        ├── train_yolo11.py           # YOLO11 training
        └── predict_yolo11.py         # YOLO11 prediction
```

## 🔧 Advanced Usage

### Custom Dataset Integration

Add your own dataset by following the structure:

```
data/your_dataset/
├── train/
│   ├── images/      # RGB images (.jpg, .png)
│   └── masks/       # Instance masks (.png, integer encoding)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Then add download logic to `datasets.sh` and run:

```bash
./datasets.sh your_dataset
```

### SAM3 Prompt Engineering

Modify prompt generation strategies in `src/advanced_prompting.py` to experiment with:
- Point prompt sampling strategies
- Bounding box vs point prompts
- Multi-prompt ensembling
- Negative prompt incorporation

### Hyperparameter Tuning

Edit training scripts to adjust:
- Learning rates (SAM3: `1e-4`, YOLO11: `1e-3`)
- Batch sizes (SAM3: 4-8, YOLO11: 8-16)
- Epochs (SAM3: 15-30, YOLO11: 50-100)
- LoRA rank (default: 8, try: 4, 16, 32)
- Data augmentation intensity

## 📊 Expected Performance

### SAM3 vs YOLO11 Comparison (MinneApple Dataset)

| Model | Training | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Speed |
|-------|----------|-------|------|-------------|----------|-------|
| **SAM3 Zero-Shot** | None | ~0.65 | ~0.75 | ~0.78 | ~0.72 | Slow |
| **SAM3 Full FT** | Full | ~0.78 | ~0.85 | ~0.87 | ~0.83 | Slow |
| **SAM3 LoRA** | Efficient | ~0.76 | ~0.84 | ~0.86 | ~0.82 | Slow |
| **YOLO11** | Standard | ~0.74 | ~0.82 | ~0.88 | ~0.77 | **Fast** |

*Note: Actual results vary by dataset, hardware, and training configuration*

### Key Findings

- ✅ **SAM3 zero-shot** achieves impressive baseline without training
- ✅ **SAM3 fine-tuning** significantly improves domain-specific performance
- ✅ **SAM3 LoRA** matches full fine-tuning with 100x fewer parameters
- ✅ **YOLO11** offers best inference speed for real-time applications
- ✅ **SAM3** excels at segmentation boundary precision
- ✅ **YOLO11** achieves higher precision with conservative predictions

## 🔍 SEO Keywords

**SAM3 fine-tuning**, **SAM3 agriculture**, **SAM3 predictions**, **SAM3 zero-shot**, **SAM3 LoRA**, **SAM3 vs YOLO11**, **segment anything model 3**, **SAM3 instance segmentation**, **SAM3 computer vision**, **agricultural AI**, **apple detection**, **weed segmentation**, **SAM3 prompt engineering**, **parameter-efficient fine-tuning**, **foundation model agriculture**, **SAM3 PyTorch**, **SAM3 training**, **SAM3 evaluation**, **SAM3 benchmarking**, **precision agriculture AI**

## 📚 Citation

If you use this framework in your research or projects, please cite:

```bibtex
@software{sam3_roumeliotis_sapkota_2025,
  author = {Roumeliotis, Konstantinos I. and Sapkota, Ranjan},
  title = {SAM3 Fine-Tuning Framework for Agricultural Instance Segmentation: A Comprehensive Comparison with YOLO11},
  year = {2025},
  institution = {University of the Peloponnese, Department of Informatics and Telecommunications, Greece and Cornell University, Biological & Environmental Engineering, Ithaca, New York, USA},
  url = {https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation}
}
```

**Note:** A research paper based on this framework is forthcoming. Please check back for updated citation information.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional agricultural datasets
- New evaluation metrics
- Prompt engineering strategies
- Model architecture variants
- Deployment optimizations

## 📄 License

This project uses a **composite license structure**:

### Original Framework Code (MIT License)
**Copyright (c) 2025 Konstantinos I. Roumeliotis (University of Peloponnese) and Ranjan Sapkota (Cornell University)**

Our original contributions (architecture, scripts, evaluation, visualization) are MIT licensed.

You are free to:
- ✅ Use this framework for commercial and non-commercial purposes
- ✅ Modify and distribute the code
- ✅ Use in private and public projects

**Requirements:**
- 📝 Include the original copyright notice and license in all copies
- 📚 Cite the authors in academic publications using this code
- 🔗 Provide attribution when redistributing or building upon this work

### Third-Party Components

This framework integrates with:
- **SAM3**: Meta's SAM License (custom research license)
- **YOLO11**: Ultralytics AGPL-3.0 (commercial license available)
- **PyTorch, Transformers, PEFT**: Apache 2.0 / BSD licenses
- **Datasets**: Subject to their respective licenses

⚠️ **Important**: For commercial use involving YOLO11, you may need an [Ultralytics Enterprise License](https://ultralytics.com/license). SAM3 usage should comply with [Meta's SAM License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).

See the [LICENSE](LICENSE) file for complete terms and third-party license details.

## 👥 Authors

**Dr. Konstantinos I. Roumeliotis**  
University of the Peloponnese  
Department of Informatics and Telecommunications  
Greece

**Dr. Ranjan Sapkota**  
Cornell University  
Biological & Environmental Engineering  
Ithaca, New York, USA

## 🙏 Acknowledgments

- **Meta AI** - SAM3 (Segment Anything Model) foundation model
- **Ultralytics** - YOLO11 implementation and training framework
- **Hugging Face** - Transformers library and model hub
- **PEFT** - Parameter-efficient fine-tuning library
- **MinneApple Dataset** - University of Minnesota
- **WeedsGalore Dataset** - Agricultural weed detection research

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Applied-AI-Research-Lab/SAM3-FineTuning-vs-YOLO11-Agriculture-Segmentation/discussions)

---

**⭐ Star this repository if you find it useful for SAM3 research or agricultural AI applications!**

**🔖 Keywords**: #SAM3 #SegmentAnything #YOLO11 #AgricultureAI #ComputerVision #InstanceSegmentation #FineTuning #LoRA #PyTorch #DeepLearning #PrecisionAgriculture
