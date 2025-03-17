# Deep-Switching-State-Space-Model

## [Deep Switching State Space Model (DS<sup>3</sup>M) for Nonlinear Time Series Forecasting with Regime Switching](https://arxiv.org/abs/2106.02329)
Authors: Xiuqin Xu, Hanqiu Peng, Ying Chen

This repository provides a PyTorch implementation of DS<sup>3</sup>M, which incorporates discrete and continuous latent variables to capture possible regime-switching behavior in time series.

## Environment Setup

1. Create and activate a conda environment:
   ```bash
   conda create -n ds3m_env python=3.10
   conda activate ds3m_env
   ```

2.	Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Structure

```bash
Deep-Switching-State-Space-Model/
│
├─ data/
│   └─ [Contains 8 different datasets]
├─ results/
│   ├─ outputs/
│   │   └─ outputs.csv
│   └─ checkpoints/
│       ├─ Electricity/
│       │   └─ checkpoint.tar
│       ├─ Lorenz/
│       │   └─ checkpoint.tar
│       ├─ Seattle/
│       │   └─ checkpoint.tar
│       ├─ Toy/
│       │   └─ checkpoint.tar
│       ├─ Hangzhou/
│       │   └─ checkpoint.tar
│       ├─ Pacific/
│       │   └─ checkpoint.tar
│       ├─ Sleep/
│       │   └─ checkpoint.tar
│       └─ Unemployment/
│           └─ checkpoint.tar
├─ figures/
│   └─ [Stores generated plots showing DS^3M performance]
├─ tables/
│   └─ [Stores generated tables showing DS^3M performance]
├─ src/
│   ├─ DSSSMCode.py 
│   └─ utils.py      
├─ LICENSE           
├─ README.md
├─ main.py
├─ table_generation.py
└─ requirements.txt
```

## Script Descriptions
- **`main.py`**  
  Parses arguments (`-p` for dataset, `--train` for training), loads data, trains or loads a model, and generates result plots.

- **`src/DSSSMCode.py`**  
  Defines the DS<sup>3</sup>M model (discrete + continuous latent variables, RNN encoders, etc.) using PyTorch. Handles forward passes, multi-step forecasting, and loss computations (KL divergence, likelihood, etc.).

- **`src/utils.py`**  
  Provides data normalization (`normalize_fit`, `normalize_invert`), dataset creation (`create_dataset2`), evaluation metrics (RMSE, MAPE), classification scores, and plotting utilities.

## Replication Steps
1. Download the checkpoints folder from [this link](https://www.dropbox.com/scl/fi/uhqpjmubfcr5wr102nxzr/checkpoints.zip?rlkey=2p6xabwl7e3325eaxefc9slbj&dl=0), unzip it, and place `checkpoints/` folder under `results/`.

2. Generate figures

Run below command one by one to generate the figures
```
python main.py -p Toy 
python main.py -p Lorenz
python main.py -p Sleep
python main.py -p Unemployment
python main.py -p Hangzhou
python main.py -p Seattle
python main.py -p Pacific
python main.py -p Electricity
```
All generated figures are saved to `figures/`. 
- **Figure 2(a)**: `Toy_Prediction.png`
- **Figure 2(b)**: `Lorenz_Prediction.png`
- **Figure 3(a)**: `Sleep_Prediction.png`
- **Figure 3(b)**: `Unemployment_Prediction.png`
- **Figure 3(c)**: `Hangzhou_Station 0.png`
- **Figure 3(d)**: `Hangzhou_Station 40.png`
- **Figure 3(e)**: `Seattle_Station 0.png`
- **Figure 3(f)**: `Seattle_Station 322.png`
- **Figure 3(g)**: `Pacific_Station 0.png`
- **Figure 3(h)**: `Pacific_Station 840.png`
- **Figure 3(i)**: `Electricity_Station 0.png`
- **Figure 3(j)**: `Electricity_Station 24.png` 

> **Note**: Due to that the forecasting results are generated via Monte Carlo method, the produced results will be slightly different with different runs.

- The specific source and description of the data can be found in the paper -- Section 4.1 (**Simulations**) and Section 4.2 (**Real data analysis**).

- By default, the script loads the pre-trained checkpoint located at `results/checkpoints/<dataset_name>/checkpoint.tar` for inference. 

If you want to retrain the model from scratch, add --train, for example:
```bash
python main.py -p Pacific --train
```
In this mode, the best checkpoint will be saved to `results/checkpoints/<dataset_name>/best.tar`.

3. Generate tables

Run below command to generate the tables
```bash
python table_generation.py
```
All generated tables will be saved to `tables/`.
- **Table 1**: `table1_part1.csv` & `table1_part2.csv`
- **Table 3**: `table3_part1.csv` & `table3_part2.csv`

This will generate the tables based on intermediate results (results/outputs/outputs.csv) obtained using a GPU/CPU.

## Citation

If you find this code useful, please cite:
```bibtex
@misc{xu2025deepswitchingstatespace,
      title={Deep Switching State Space Model (DS$^3$M) for Nonlinear Time Series Forecasting with Regime Switching}, 
      author={Xiuqin Xu and Hanqiu Peng and Ying Chen},
      year={2025},
      eprint={2106.02329},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2106.02329}, 
}
```

## Reproducibility Package Information

**Assembled Date:** March 4, 2025  
**Package Authors:** Xiuqin Xu and Hanqiu Peng  
**Contact Email:** [xiuqin.sherry.xu@gmail.com](mailto:xiuqin.sherry.xu@gmail.com)