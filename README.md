# 📌 Adaptive Hybrid Recommender System for Requirements Reuse

## 🔍 Project Overview

This project implements a hybrid recommender system tailored for software requirements engineering. Designed for seamless integration into existing workflows, it treats stakeholders as users and requirements as recommendable items. Once a stakeholder interacts with at least two requirements, the system generates personalized predictions to suggest the most relevant items based on inferred preferences.

---

## 🔧 Key Features

- Personalized requirement recommendations using collaborative and content-based filtering
- Lightweight integration with minimal disruption to engineering processes
- Offline semantic similarity computation using GloVe vectors
- Domain-specific TF-IDF scoring based on the PURE corpus
- Scalable and sustainable update cycle aligned with engineering iterations
---

## 🛠️ Maintenance Workflow

- Regenerate semantic similarity matrix post each engineering cycle
- Recalculate TF-IDF scores to reflect evolving project contexts
- All updates are computationally efficient and infrastructure-friendly

---

# 🚀 Running and Reproducing Experiments

The `main.py` script serves as the **single entry point** to reproduce all experiments reported in the paper  
**"An Adaptive Hybrid Recommender System for Requirements Reuse"**.

### Quick Start
Run the following command:
```bash
python main.py
```

Running this command will:
- Execute the **proposed adaptive hybrid method**
- Run all **state-of-the-art baseline methods**
- Automatically generate all **evaluation datasets**
- Produce **Excel** result files summarizing performance

---

## 🗂 Input / Output Conventions

### Input Directory
`input_output_data/`  
- The system reads input (incomplete) datasets from this folder.  
- To use your own data, remove all existing files in this directory and place your dataset files there (keeping the same format).

### Automatic Dataset Generation
The tool automatically produces **7 incomplete datasets** per scenario level for each of the following scenarios:
- `sparsity` (sp)
- `user cold-start` (ucs)
- `item cold-start` (ics)

This process repeats for **3 independent trials**, yielding:
**7 datasets × 3 scenarios × 3 trials = 63 generated dataset files**

### Evaluation Output
Results are saved as **Excel files** using the naming pattern:
`results_tr[x]_[y].xlsx`

Where:
- `[x]` ∈ {1, 2, 3} — trial number  
- `[y]` ∈ {sp, ucs, ics} — evaluation scenario  

**Example files:**
- `results_tr1_sp.xlsx`  
- `results_tr2_ucs.xlsx`  
- `results_tr3_ics.xlsx`  

Each file contains detailed metrics for all methods across trials and scenarios.


## ▶️ How to Run

### Reproduce All Experiments
```bash
python main.py
```

### Run with new Data
```bash
rm input_output_data/*
python main.py
```

The system will regenerate all incomplete datasets (7 per scenario × 3 scenarios × 3 trials = 63 total)  
and export the results as Excel files using the naming protocol above.
