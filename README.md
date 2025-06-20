# Pattern-Based Enhancement for Model Transformations using Neural Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-NIER%202025-green.svg)](https://github.com/gamou67/trans-sem-preservation.git)

This repository contains the implementation of our semantic preservation enhancement framework for model transformations, combining transformer-based neural embeddings with automated pattern-based generation.

## 📋 Overview

Our approach addresses the critical challenge of semantic preservation in cross-metamodel transformations by:

- **Neural Semantic Assessment**: Using DistilBERT embeddings to measure semantic similarities with 96.5% average neural BAS
- **Automated Pattern Application**: Two validated preservation patterns achieving up to 104.8% improvement
- **Large-Scale Validation**: Evaluation on 266 real transformations with 95.9% success rate
- **Industrial Applicability**: Processing times <0.3s per transformation with enhanced scalability

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/gamou67/trans-sem-preservation.git
cd trans-sem-preservation
pip install -r requirements.txt
```

### Basic Usage

Run the main semantic preservation framework:

```bash
streamlit run semantic_framework_final.py
```

Generate analysis graphs from results:

```bash
python real_data_graphs_final.py
```

## 🔬 Dataset

This project uses the **ModelSet** dataset, a comprehensive collection of models for machine learning in model-driven engineering.

- **Source**: [ModelSet on Figshare](https://figshare.com/s/5a6c02fa8ed20782935c?file=24495371)
- **Size**: 266 transformations across 5 transformation types
- **Coverage**: 2,489 token pairs extracted and analyzed
- **Types**: UML→Ecore, Ecore→Java, UML→Java, Ecore→EcoreV2, BPMN→PetriNet

The `modelset/` directory should contain the extracted dataset files.
