# Pattern-Based Enhancement for Model Transformations using Neural Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ASE%202025-green.svg)](https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings)

This repository contains the implementation of our semantic preservation enhancement framework for model transformations, combining transformer-based neural embeddings with automated pattern-based generation.

## 📋 Overview

Our approach addresses the critical challenge of semantic preservation in cross-metamodel transformations by:

- **Neural Semantic Assessment**: Using DistilBERT embeddings to measure semantic similarities
- **Automated Pattern Application**: Three validated preservation patterns for gap mitigation
- **Large-Scale Validation**: Evaluation on 149 real transformations with 94.6% success rate
- **Industrial Applicability**: Processing times <0.6s per transformation

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings.git
cd pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings
pip install -r requirements.txt
```

### Basic Usage

Run the main semantic preservation framework:

```bash
python semantic_framework_preservation.py
```

Generate analysis graphs from results:

```bash
python real_data_graphs.py
```

## 📁 Project Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── semantic_framework_preservation.py # Main framework implementation
├── real_data_graphs.py               # Graph generation from results
├── modelset/                          # ModelSet dataset directory
│   ├── UML_models/                   # UML source models
│   ├── Ecore_models/                 # Ecore models
│   ├── Java_models/                  # Java target models
│   └── BPMN_models/                  # BPMN process models
├── results/                          # Generated results and evaluations
├── patterns/                         # Pattern implementation modules
└── docs/                            # Documentation and paper materials
```

## 🔬 Dataset

This project uses the **ModelSet** dataset, a comprehensive collection of models for machine learning in model-driven engineering.

- **Source**: [ModelSet on Figshare](https://figshare.com/s/5a6c02fa8ed20782935c?file=24495371)
- **Size**: 149 transformations across 5 transformation types
- **Coverage**: 1,957 model elements analyzed
- **Types**: UML→Ecore, Ecore→Java, UML→Java, Ecore→EcoreV2, BPMN→PetriNet

The `modelset/` directory should contain the extracted dataset files.

## 🧪 Running Experiments

### Full Evaluation Pipeline

```bash
# Run complete evaluation on all 149 transformations
python semantic_framework_preservation.py --mode full_evaluation

# Run evaluation on specific transformation type
python semantic_framework_preservation.py --type UML_to_Ecore

# Run with custom threshold
python semantic_framework_preservation.py --similarity_threshold 0.4
```

### Generate Research Graphs

```bash
# Generate all paper figures
python real_data_graphs.py --generate_all

# Generate specific figure
python real_data_graphs.py --figure distribution_analysis
python real_data_graphs.py --figure pattern_effectiveness
python real_data_graphs.py --figure transformation_comparison
python real_data_graphs.py --figure gaps_coverage
```

## 📊 Results

Our evaluation demonstrates significant improvements in semantic preservation:

| Metric                        | Value  |
| ----------------------------- | ------ |
| **Success Rate**              | 94.6%  |
| **Average Improvement**       | +2.81% |
| **Effect Size (Cohen's d)**   | 2.026  |
| **Transformations Evaluated** | 149    |
| **Elements Analyzed**         | 1,957  |

### Pattern Effectiveness

| Pattern Type                    | Applications | Avg Improvement |
| ------------------------------- | ------------ | --------------- |
| **MetadataPreservationPattern** | 137 (71.4%)  | +2.78%          |
| **BehavioralEncodingPattern**   | 54 (28.1%)   | +3.30%          |
| **HybridPattern**               | 1 (0.5%)     | +3.20%          |

## 🔧 Configuration

### Framework Parameters

Key configuration options in `semantic_framework_preservation.py`:

```python
# Neural embedding configuration
EMBEDDING_MODEL = "distilbert-base-uncased"
SIMILARITY_THRESHOLD = 0.4
BATCH_SIZE = 32

# Pattern application settings
IMPROVEMENT_THRESHOLD = 0.1
MAX_PATTERNS_PER_GAP = 3

# Evaluation settings
CONFIDENCE_LEVEL = 0.95
STATISTICAL_SIGNIFICANCE = 0.001
```

### Supported Transformation Types

- ✅ **UML→Ecore**: 100% success rate, +3.52% avg improvement
- ✅ **Ecore→Java**: 100% success rate, +2.96% avg improvement
- ✅ **UML→Java**: 96% success rate, +3.04% avg improvement
- ✅ **Ecore→EcoreV2**: 100% success rate, +1.49% avg improvement
- ❌ **BPMN→PetriNet**: 0% success rate (domain-specific limitations)

## 📈 Analysis and Visualization

The `real_data_graphs.py` script generates publication-ready figures:

1. **Distribution Analysis**: Histogram and box plots of improvements
2. **Transformation Comparison**: BA score improvements by type
3. **Pattern Effectiveness**: Usage distribution and effectiveness
4. **Gap Coverage**: Detection vs treatment analysis

## 🏗️ Architecture

### Core Components

1. **Token Pair Extraction**: Model element to meta-element relationship mapping
2. **Neural Embedding Engine**: DistilBERT-based semantic similarity computation
3. **Gap Detection**: Threshold-based identification of preservation failures
4. **Pattern Application**: Automated selection and application of preservation strategies
5. **Evaluation Framework**: Statistical validation and result analysis

### Pattern Categories

1. **MetadataPreservationPattern**: Annotation-based preservation for lost metadata
2. **BehavioralEncodingPattern**: Behavioral semantics encoding in target constructs
3. **HybridPattern**: Combined strategies for complex preservation scenarios

## 🚀 Performance

- **Processing Time**: <0.6s per transformation (average)
- **Memory Usage**: ~2GB for full dataset evaluation
- **Scalability**: Tested up to 150 elements per model
- **Accuracy**: 94.6% success rate across diverse transformation types

## 📝 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{elbachir2025semantic,
  title={Enhancing Semantic Preservation in Model Transformations through Pattern-Based Generation: A Large-Scale Empirical Study},
  author={Anonymous Authors},
  booktitle={Proceedings of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year={2025},
  organization={IEEE}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [ASE 2025 Submission](https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings)
- **Dataset**: [ModelSet on Figshare](https://figshare.com/s/5a6c02fa8ed20782935c?file=24495371)
- **Documentation**: [GitHub Pages](https://elbachir67.github.io/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings/)

## 🆘 Support

For questions and support:

- **Issues**: [GitHub Issues](https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings/issues)
- **Discussions**: [GitHub Discussions](https://github.com/elbachir67/pattern-Based-Enhancement-for-Model-Transformations-using-Neural-Embeddings/discussions)
- **Email**: [Contact the authors](mailto:authors@university.edu)

## 🏆 Acknowledgments

- ModelSet consortium for providing the comprehensive model repository
- HuggingFace for the Transformers library and pre-trained models
- The Model-Driven Engineering community for valuable feedback

---

**Keywords**: Model Transformations, Semantic Preservation, Neural Embeddings, Pattern-Based Generation, Cross-Metamodel Transformations, DistilBERT, Empirical Validation
