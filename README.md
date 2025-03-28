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
├─ data/
│   └─ [Contains 8 different datasets]
├─ results/
│   ├─ outputs/
│   │   ├─ outputs.csv               # Default outputs from pre-trained models
│   │   └─ outputs_generated.csv     # Outputs generated after retraining models
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
│   └─ [Stores generated plots showing DS³M performance]
├─ tables/
│   └─ [Stores generated tables showing DS³M performance]
├─ src/
│   ├─ DSSSMCode.py   # DS³M model implementation and forecasting routines
│   └─ utils.py       # Data preprocessing, evaluation metrics, and helper functions
├─ LICENSE
├─ README.md
├─ main.py        
├─ reload_experiment.sh  # Script to run pre-trained experiments (Option 1)
├─ requirements.txt
├─ retrain_experiment.sh # Script to retrain models from scratch (Option 2)
└─ table_generation.py   # Script to generate tables based on model outputs  
```
- Download the checkpoints folder from [this link](https://www.dropbox.com/scl/fi/uhqpjmubfcr5wr102nxzr/checkpoints.zip?rlkey=2p6xabwl7e3325eaxefc9slbj&dl=0), unzip it, and place `checkpoints/` folder under `results/`.
- The specific source and description of the data can be found in the paper -- Section 4.1 (**Simulations**) and Section 4.2 (**Real data analysis**).

## Script Descriptions
- **`main.py`**  
  Parses arguments (`-p` for dataset, `--train` for training), loads data, trains or loads a model, and generates result plots.

- **`src/DSSSMCode.py`**  
  Defines the DS<sup>3</sup>M model (discrete + continuous latent variables, RNN encoders, etc.) using PyTorch. Handles forward passes, multi-step forecasting, and loss computations (KL divergence, likelihood, etc.).

- **`src/utils.py`**  
  Provides data normalization (`normalize_fit`, `normalize_invert`), dataset creation (`create_dataset2`), evaluation metrics (RMSE, MAPE), classification scores, and plotting utilities.

## Code usage:
1. To load the checkpoint in `results/checkpoints/<dataset_name>/checkpoint.tar`
```bash
python main.py -p Pacific
```
2. You want to retrain the model from scratch, add --train, for example:
```bash
python main.py -p Pacific --train
```
The best checkpoint will be saved to `results/checkpoints/<dataset_name>/best.tar`.
> You can replace `{dataset_name}` in `-p {dataset_name}` to other dataset (Toy, Lorenz, Sleep, Unemployment, Hangzhou, Seattle,  Pacific, Electricity).

## Replication Steps
### Option 1. To load the pre-trained checkpoints
1. Download the checkpoints folder from [this link](https://www.dropbox.com/scl/fi/uhqpjmubfcr5wr102nxzr/checkpoints.zip?rlkey=2p6xabwl7e3325eaxefc9slbj&dl=0) as described above.
2. Run the following command in the terminal to obtain the figures and tables:

```bash
chmod +x reload_experiment.sh  
./reload_experiment.sh
```
- Inside `reload_experiment.sh`, we will load the pre-trained for each dataset.
> **Note**: Since forecasting results are generated using the Monte Carlo method, the results will vary slightly between runs.

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

All generated tables will be saved to `tables/`. It will also be shown in the terminal. This will generate the tables based on saved results (results/outputs/outputs.csv).
- **Table 1**: `table1_part1.csv` & `table1_part2.csv`
- **Table 3**: `table3_part1.csv` & `table3_part2.csv`

### Option 2. To retrain the models and obtain the prediction results for each dataset
Run below command in terminal
```bash
chmod +x retrain_experiment.sh  
./retrain_experiment.sh
```
Inside `retrain_experiment.sh`, we retrain the model for for each dataset. This process will take approximately 2 hours on a MacBook Pro with an M1 chip and 16GB of memory.

All generated figures are saved in the `figures/` directory similar to above.

All generated tables will be saved in the `tables/` directory similar to above. It will also be shown in the terminal.

> **Note**: Since the neural networks are trained using SGD (which can be unstable for some datasets) and forecasting results are generated via Monte Carlo methods, the produced results may vary slightly between runs. In cases where noticeable deviations occur, you may need to retrain the models.

## Additional Models

The current repository includes only the implementation of the DS³M model developed by our team. To reproduce the complete results presented in Tables 1 and 3 of our paper, please adapt the implementations of other  referenced models to our use case.:

- **DSARF**: [https://github.com/ostadabbas/DSARF](https://github.com/ostadabbas/DSARF)
- **SNLDS**: [https://github.com/google-research/google-research/tree/master/snlds](https://github.com/google-research/google-research/tree/master/snlds)
- **SRNN**: [https://github.com/marcofraccaro/srnn](https://github.com/marcofraccaro/srnn)

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
