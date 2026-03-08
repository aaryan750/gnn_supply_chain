# Contagion Alpha: Predicting Equity Returns via Global Supply Chain Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Executive Summary
This repository contains the codebase for the research paper: *"Extracting Contagion Alpha: Predicting Equity Cross-Sectional Returns via Global Supply Chain Graph Neural Networks"*.

Traditional quantitative models (like Fama-French or standard momentum) evaluate equities in isolation. However, the global economy is deeply interconnected. A supply chain shock in semiconductor manufacturing propagates through the global network, impacting auto manufacturers, consumer electronics, and logistics firms. 

This project proves that **Graph Convolutional Networks (GCNs)** can model these non-linear, upstream-downstream relationships to generate predictive alpha that traditional models miss.

## 🏗️ Project Architecture
We formulate the equity market as a Graph $G=(V, E)$:
*   **Nodes (V):** S&P 500 Companies.
*   **Node Features:** Historical 5-day return, volatility, and volume momentum.
*   **Edges (E):** Industry correlation adjacency matrix (a proxy for supply chain relationships).
*   **Target:** Next-day cross-sectional return prediction.

Our baseline model is a standard Multi-Layer Perceptron (MLP). Our core model is a Graph Convolutional Network (GCN) that outperforms the baseline by aggregating feature data from neighboring nodes (suppliers/customers).

## 📂 Directory Structure
```text
gnn_supply_chain/
├── data/
│   ├── raw/                 # OHLCV data fetched via yfinance
│   └── processed/           # Adjacency matrices and node features
├── models/
│   ├── baseline_mlp.py      # Standard Neural Net (Baseline)
│   └── supply_chain_gcn.py  # Graph Convolutional Network (Our Model)
├── notebooks/
│   ├── 1_Data_Exploration_and_Graph_Building.ipynb
│   └── 2_Model_Training_and_Backtesting.ipynb
├── src/
│   ├── data_loader.py       # yfinance fetching engine
│   ├── build_graph.py       # NetworkX graph construction
│   └── backtester.py        # Vectorized long/short strategy backtester
├── requirements.txt
└── README.md
```

## 🚀 Quick Start
### 1. Installation
Clone the repo and install the required PyTorch Geometric environment:
```bash
git clone https://github.com/your-username/gnn_supply_chain.git
cd gnn_supply_chain
python -m venv venv
### Activate venv
pip install -r requirements.txt
```

### 2. Running the Pipeline
To reproduce the research results:
1. Fetch S&P 500 features: `python src/data_loader.py`
2. Build the Adjacency Matrix: `python src/build_graph.py`
3. Run training and backtesting:
   * interactively using the notebook at `notebooks/2_Model_Training_and_Backtesting.ipynb`, or
   * from the command line with real data: `python train_script.py` (after data has been fetched).

## 📊 Results Summary
Using a dollar-neutral, market-neutral simulated out-of-sample backtest going long the top prediction decile and shorting the bottom decile on the live S&P 500 feature dataset:
*   **Annualized Return:** 4.07%
*   **Annualized Volatility:** 5.06%
*   **Sharpe Ratio:** 0.80
*   **Max Drawdown:** -1.38%

---
*For the complete mathematical formulation and literature review, please read the accompanying [Research Paper PDF](#).*
