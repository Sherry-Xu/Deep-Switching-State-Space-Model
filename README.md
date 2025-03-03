# Deep-Switching-State-Space-Model

## [Deep Switching State Space Model (DS^3M)](https://arxiv.org/abs/2106.02329)
Authors: Xiuqin Xu, Hanqiu Peng, Ying Chen

This repository provides a PyTorch implementation of DS^3M, which incorporates discrete and continuous latent variables to capture possible regime-switching behavior in time series. The code covers model training, inference, and visualization across multiple real-world datasets.

---

## Environment Setup

1. Create and activate a conda environment:
   ```bash
   conda create -n ds3m_env python=3.10
   conda activate ds3m_env
   ```

2.	Install required packages:

pip install -r requirements.txt

**Repository Structure**

```bash
Deep-Switching-State-Space-Model/
│
├─ data/
│   └─ [Contains 8 different datasets for training, validation, and testing]
├─ results/
│   ├─ figures/
│   │   └─ [Stores generated plots showing DS^3M performance]
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
├─ src/
│   ├─ DSSSMCode.py 
│   └─ utils.py      
├─ main.py           
├─ README.md
└─ requirements.txt
```

To reproduce results, download the checkpoints from [this link](https://www.dropbox.com/scl/fo/fytbaft45895y3057r8dp/AIh1K1ZKmI3C8nUuRgZ9wbY?rlkey=vovaus04jfeiqk2gqkw3nnpbb&dl=0), unzip them, and place them under `results/checkpoints/`.

Usage

From the command line, run:

python main.py -p Electricity

Replace Electricity with other dataset names (e.g., Toy, Lorenz, Sleep, Unemployment, Hangzhou, Seattle, Pacific) to test those datasets. By default, the script loads the pre-trained checkpoint for inference. If you want to retrain from scratch, add --train, for example:

python main.py -p Toy --train

It will save the best checkpoint (best.tar) to results/checkpoints/<dataset_name>.

Key Scripts
	•	main.py
Parses arguments (-p for dataset, -train for training), loads data, trains or loads a model, and generates result plots.
	•	src/DSSSMCode.py
Defines the DS^3M model (discrete + continuous latent variables, RNN encoders, etc.) using PyTorch. Handles forward passes, multi-step forecasting, and loss computations (KL divergence, likelihood, etc.).
	•	src/utils.py
Provides data normalization (normalize_fit, normalize_invert), dataset creation (create_dataset2), evaluation metrics (RMSE, MAPE), classification scores, and plotting utilities.

Figures

All generated figures are saved to results/figures/. The following mappings show how each figure corresponds to the paper:
	•	Figure 2(a): Toy_Prediction.png
	•	Figure 2(b): Lorenz_Prediction.png
	•	Figure 3(a): Sleep_Prediction.png
	•	Figure 3(b): Unemployment_Prediction.png
	•	Figure 3(c): Hangzhou_Station 0.png
	•	Figure 3(d): Hangzhou_Station 40.png
	•	Figure 3(e): Seattle_Station 0.png
	•	Figure 3(f): Seattle_Station 322.png
	•	Figure 3(g): Pacific_Station 0.png
	•	Figure 3(h): Pacific_Station 840.png
	•	Figure 3(i): Electricity_Station 0.png
	•	Figure 3(j): Electricity_Station 24.png

Citation

If you find this code useful, please cite:

@article{xu2021deep,
  title={Deep Switching State Space Model (DS \$^3\$ M) for Nonlinear Time Series Forecasting with Regime Switching},
  author={Xu, Xiuqin and Peng, Hanqiu and Chen, Ying},
  journal={arXiv preprint arXiv:2106.02329},
  year={2021}
}
