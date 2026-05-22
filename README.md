# MGSGC: Multi-Granularity Spectral Graph Coarsening

This repository contains the official implementation of **MGSGC** (Multi-Granularity Spectral Graph Coarsening), a novel framework designed to simplify large-scale graph representations while preserving both global spectral properties and local structural details.

---

## 📖 Abstract

Graph coarsening is the process of simplifying large-scale graph representations while preserving essential structural characteristics to improve computational efficiency in graph processing. Conventional techniques primarily reduce graph size through node and edge merging but often inadequately preserve both global spectral properties and local structural details. Real-world graphs exhibit inherent hierarchical complexity comprising critical global topological patterns and local structural features necessary for accurate analysis. To address these limitations, we present a **Multi-Granularity Spectral Graph Coarsening (MGSGC)** framework that systematically integrates spectral graph analysis with local structural preservation through multi-granularity operations. Our approach initiates with hierarchical graph decomposition, where node merging generates structurally homogeneous subgraphs. Spectral analysis of normalized Laplacian matrices guides iterative coarsening optimization, using spectral distance metrics to identify subgraphs requiring refinement. A dual-resolution mechanism preserves global spectral signatures and local connectivity patterns simultaneously, ensuring retention of both macroscopic and microscopic structural information. Comprehensive experiments across multiple benchmark datasets demonstrate that MGSGC outperforms recent methods, achieving higher accuracy, superior structural preservation, and strong resilience to label noise, ensuring robust performance in real-world scenarios.

---


## 🛠️ Requirements & Installation

The code has been verified in a Linux environment with **Python 3.10.9**, **PyTorch 1.13.1**, and **CUDA 11.6** capabilities.

### Tested Environment
* **OS**: Linux (Ubuntu)
* **Python**: 3.10.9
* **PyTorch**: 1.13.1
* **CUDA**: 11.6

### Key Dependencies
Based on the project's environment, the following core packages are required:
* networkx == 2.5
* numpy == 1.23.5
* scipy == 1.10.0
* scikit-learn == 1.2.1
* netlsd (For network signature extraction)

### Installation
You can configure your environment and install the core dependencies using pip:
```bash
pip install networkx==2.5 numpy==1.23.5 scipy==1.10.0 scikit-learn==1.2.1 netlsd


---


🚀 Usage

Run Experiments
Execute the main script to start the multi-granularity graph coarsening and downstream classification evaluation:

python main.py 


---


✒️ Citation
If you find this work, code, or framework useful for your research, please consider citing our paper published in Information Sciences:

@article{NI2026122748,
  title     = {Multi-granularity spectral graph coarsening},
  journal   = {Information Sciences},
  volume    = {726},
  pages     = {122748},
  year      = {2026},
  issn      = {0020-0255},
  doi       = {https://doi.org/10.1016/j.ins.2025.122748},
  author    = {Jinyuan Ni and Long Chen and Ning Yu},
  keywords  = {Graph coarsening, Spectral information, Multi-granularity, Graph decomposition}
}