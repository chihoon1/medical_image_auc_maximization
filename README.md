# Deep AUC maximization on MedMNIST

This repo compares LibAUC's AUC-maximization losses with standard cross-entropy training on MedMNIST classification tasks. It started as a machine learning course project at Texas A&M (CSE633, Spring 2023). Four of the notebooks were later reworked with a clearer baseline, more metrics and saved outputs.

## Main notebooks

| Dataset | Task | Compared | Headline result (test ROC-AUC) | Notebook |
|---|---|---|---|---|
| ChestMNIST | 14 findings, multi-label, 28x28 X-rays | BCE baseline, pos_weight BCE, Focal Loss, LibAUC AUC-margin loss | pos_weight BCE 0.769, Baseline 0.742, Focal Loss 0.726, AUC-margin 0.633 (macro) | [chestMNIST.ipynb](chestMNIST.ipynb) |
| SynapseMNIST3D | inhibitory vs. excitatory synapse, 28x28x28 volumes | BCE baseline, four LibAUC augmentation sets, 3D convolution, Optuna | Baseline 0.626, tuned Conv3d 0.778 | [SynapseMNIST.ipynb](SynapseMNIST.ipynb) |
| VesselMNIST3D | aneurysm vs. healthy, 28x28x28 volumes | BCE baseline, LibAUC with and without balancing, 3D convolution, Optuna | Baseline 0.929, hand-picked LibAUC Conv3d 0.928 (tuned arm not recorded) | [VesselDataset.ipynb](VesselDataset.ipynb) |
| BreastMNIST | malignant vs. normal/benign ultrasound, 28x28 | BCE baseline, four LibAUC augmentation sets, Optuna | Baseline 0.813, tuned 0.876, best hand-picked 0.899 | [BreastDataset.ipynb](BreastDataset.ipynb) |

Chest is the only one with bootstrap intervals and per-label thresholds tuned on validation. Synapse and Vessel also compare 2.5D ResNets (depth slices as channels) with a 3D convolutional network. Breast is the smallest, with 156 test images.

## What the experiments show

The AUC-margin loss did not clearly beat cross-entropy in these runs. On chest, pos_weight BCE was best and the AUC-margin loss came last, though that arm used untuned hyperparameters and a different optimizer. On vessel, the BCE baseline tied the hand-picked LibAUC Conv3d (0.929 vs. 0.928). On synapse the tuned LibAUC Conv3d has the highest ROC-AUC (0.778) and 3D rotation is in every arm that scores 0.74 or higher. On breast every hand-picked LibAUC model beats the baseline on ROC-AUC, but the arms differ in stem, loss, optimizer, sampler and augmentation, so it compares recipes and not losses. The test sets are small, so differences of a few hundredths may be noise.

## How the experiments are set up

- A BCE baseline (ResNet-18, Adam, no augmentation) in every notebook.
- A hand-picked LibAUC recipe with fixed hyperparameters: `PESG` with `AUCMLoss_V2` and `DualSampler` for the binary tasks, `MultiLabelAUCMLoss` for chest.
- A 30-trial Optuna search on the hand-picked recipe in Breast, Synapse and Vessel. Chest has no search but adds pos_weight BCE and Focal Loss.
- Up to 50 epochs, with the best-validation checkpoint restored.
- Several metrics (ROC-AUC, PR-AUC, balanced accuracy, MCC, F1 or recall) and no single winner. Only some arms have every metric, and each Results section says which.

Training and evaluation helpers, and the Optuna helpers, are shared code (see below).

## Earlier notebooks

[PneumoniaDataset.ipynb](PneumoniaDataset.ipynb), [noduleDataset.ipynb](noduleDataset.ipynb) and [AdrenalDataset.ipynb](AdrenalDataset.ipynb) are earlier coursework that was not revisited.

## Repo contents

- `chestMNIST.ipynb`, `SynapseMNIST.ipynb`, `VesselDataset.ipynb`, `BreastDataset.ipynb`: the four main notebooks.
- `PneumoniaDataset.ipynb`, `noduleDataset.ipynb`, `AdrenalDataset.ipynb`: earlier coursework.
- `dataCentric_functions.py`: training loops (`train`, `multilabel_train`, `train_with_logits_loss`), a weighted sampler, on-the-fly augmentation, Gaussian noise and a validation AUC helper.
- `hyperparameter_tuning.py`: Optuna search and retrain helpers shared by the Breast, Synapse and Vessel notebooks.
- `pyproject.toml`: dependencies, split into two extras.

## Setup

The notebooks were written and run on Google Colab with a GPU (an L4 in the saved outputs). To run them locally, use [uv](https://docs.astral.sh/uv/). The four notebooks need two different LibAUC versions, which cannot share one environment, so there are two extras:

- `binary` (libauc 1.2.0, MONAI, Optuna) for `BreastDataset.ipynb`, `SynapseMNIST.ipynb` and `VesselDataset.ipynb`.
- `chest` (libauc 2.0.1 or later) for `chestMNIST.ipynb`, which needs `MultiLabelAUCMLoss`. libauc 1.2.0 does not have it.

```
uv sync --extra binary      # or: uv sync --extra chest
uv run --extra binary jupyter lab
```

Open the notebook from the repo root, so `dataCentric_functions.py` can be imported. Skip the `!pip install` cells, since uv already installed the packages.

The chest, Synapse and Vessel notebooks also have a cell that downloads the dataset with `wget` into `/root/.medmnist`, which is a Colab path. Skip that cell locally. The dataset cell calls `DataClass(..., download=True)`, and medmnist then downloads the file into `~/.medmnist` by itself. The notebooks fall back to CPU without CUDA, but that has not been tried and training would be slow.

## References

- Yang et al., MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification, Scientific Data 10, 41 (2023). https://doi.org/10.1038/s41597-022-01721-8
- Yuan et al., LibAUC: A Deep Learning Library for X-Risk Optimization, 29th SIGKDD Conference on Knowledge Discovery and Data Mining (2023).
