# Random Logit Scaling (RLS)

This repository contains the source code for the attacks and defenses used in our experiments. For the attack implementations, we used [BlackboxBench](https://github.com/SCLBD/BlackboxBench) and modified its source code to meet our needs, specifically, we implemented our own defense, along with the state-of-the-art randomized defenses used for comparison, since BlackboxBench only contains code for attacks and not defenses.

This work has been published in **Transactions on Machine Learning Research (TMLR), 2026**. See the [Citation](#citation) section below for details.

## Requirements

This code was tested with **Python 3.10**. Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Table of Contents

- [Running Experiments on CIFAR-10](#running-experiments-on-cifar-10)
  - [Step 1: Configuring the Defenses](#step-1-configuring-the-defenses)
  - [Step 2: Configuring the Attacks](#step-2-configuring-the-attacks)
  - [Step 3: Running the Attack](#step-3-running-the-attack)
- [Running Experiments on ImageNet](#running-experiments-on-imagenet)
  - [Step 1: Configuring the Defenses](#step-1-configuring-the-defenses-1)
  - [Step 2: Configuring the Attacks](#step-2-configuring-the-attacks-1)
  - [Step 3: Running the Attack](#step-3-running-the-attack-1)
- [Adaptive Attack against AAA-Sine](#adaptive-attack-against-aaa-sine)

## Running Experiments on CIFAR-10

We have augmented BlackboxBench with the source code of WideResNet-28, VGG-16, and ResNet-18, used for training our CIFAR-10 classifiers (see the `./models/` directory). Code has been added to implement the iRND, oRND, AAA, RFD, and RLS defenses for each of these models.

### Step 1: Configuring the Defenses

`./config-jsons/defense_config.json` contains the configuration for the defenses and is used to specify which defense the victim model should use. To activate a defense, modify the `defense` attribute in the JSON file (lines 2-3):

```json
"_comment1": "irnd, rfd, ornd, rls, aaa, none",
"defense": "none",
```

For instance, to activate iRND, this should be changed to:

```json
"_comment1": "irnd, rfd, ornd, rls, aaa, none",
"defense": "irnd",
```

The config file then continues with additional attributes for setting the hyperparameter values of each defense:

```json
{
    "_comment1": "irnd, rfd, ornd, rls, aaa, none",
    "defense": "none",
    "irnd": {
        "sigma_noise": 0.01
    },
    "rfd": {
        "sigma_noise": 0.7,
        "target_layers": [-1]
    },
    "ornd": {
        "sigma_noise": 1
    },
    "rls": {
        "distribution": "uniform",
        "uniform": {
            "low": 0.5,
            "high": 10
        },
        "gaussian": {
            "low": 0.2,
            "mean": 1,
            "std": 100
        }
    },
    "aaa": {
        "attractor_interval": 4,
        "calibration_loss_weight": 5,
        "dev": 0.5,
        "temperature": 1,
        "optimizer_lr": 0.1,
        "num_iter": 100
    },
    "none": ""
}
```

For example, to use iRND with *v = 0.02*, set `defense` to `"irnd"` and `sigma_noise` under `irnd` to `0.02`, as shown below:

```json
{
    "_comment1": "irnd, rfd, ornd, rls, aaa, none",
    "defense": "irnd",
    "irnd": {
        "sigma_noise": 0.02
    },
    "rfd": {
        "sigma_noise": 0.7,
        "target_layers": [-1]
    },
    "ornd": {
        "sigma_noise": 1
    },
    "rls": {
        "distribution": "uniform",
        "uniform": {
            "low": 0.5,
            "high": 10
        },
        "gaussian": {
            "low": 0.2,
            "mean": 1,
            "std": 100
        }
    },
    "aaa": {
        "attractor_interval": 4,
        "calibration_loss_weight": 5,
        "dev": 0.5,
        "temperature": 1,
        "optimizer_lr": 0.1,
        "num_iter": 100
    },
    "none": ""
}
```

#### RFD Configurations

The exact settings we use for the RFD defense are as follows for each model (we target the penultimate layer to adhere to the optimal configuration specified by the authors):

| Model | `sigma_noise` | `target_layers` |
|---|---|---|
| VGG-16 | 0.35 | `[28]` |
| ResNet-18 | 2.5 | `[4]` |
| WideResNet-28-10 | 0.7 | `[3]` |
| ResNet-50 | 3.0 | `[4]` |

For example, for VGG-16:

```json
"rfd": {
    "sigma_noise": 0.35,
    "target_layers": [28]
}
```

### Step 2: Configuring the Attacks

For each black-box attack, a JSON file is provided in the `./config-jsons/` directory to configure the hyperparameter values for that specific attack.

**For CIFAR-10 in L∞:**
```
./config-jsons/cifar10_bandit_linf_config.json
./config-jsons/cifar10_nes_linf_config.json
./config-jsons/cifar10_sign_linf_config.json
./config-jsons/cifar10_square_linf_config.json
./config-jsons/cifar10_zosignsgd_linf_config.json
```

**For CIFAR-10 in L2:**
```
./config-jsons/cifar10_bandit_l2_config.json
./config-jsons/cifar10_nes_l2_config.json
./config-jsons/cifar10_square_l2_config.json
./config-jsons/cifar10_zosignsgd_l2_config.json
```

For instance, the JSON file for SignHunter in L∞ is located at `./config-jsons/cifar10_sign_linf_config.json` and looks like this:

```json
{
  "_comment1": "===== DATASET CONFIGURATION =====",
  "dset_name": "cifar10",
  "dset_config": {},
  "_comment2": "===== EVAL CONFIGURATION =====",
  "num_eval_examples": 1000,
  "_comment3": "===== ADVERSARIAL EXAMPLES CONFIGURATION =====",
  "attack_name": "SignAttack",
  "attack_config": {
    "batch_size": 1000,
    "name": "Sign",
    "epsilon": 12.75,
    "p": "inf",
    "fd_eta": 12.75,
    "max_loss_queries": 10000
  },
  "device": "/gpu:1",
  "modeln": "resnet",
  "target": false,
  "target_type": "median",
  "seed": 123
}
```

Common among all of the config files above is the `modeln` attribute, which specifies the victim model. For experiments on CIFAR-10, it can take the following values:

| `modeln` | Model |
|---|---|
| `resnet` | ResNet-18 |
| `vgg` | VGG-16 |
| `wrn` | WideResNet-28-10 |

### Step 3: Running the Attack

Run the attack with the following command:

```bash
python attack_cifar10.py [path_to_attack_config]
```

For instance:

```bash
python attack_cifar10.py ./config-jsons/cifar10_bandit_linf_config.json
```

## Running Experiments on ImageNet

Our implementation for evaluations on ImageNet uses Torchvision's pretrained ResNet-50. Defense implementations for ResNet-50 are attached to the pretrained model using PyTorch hooks (see `./utils/model_loader.py`).

### Step 1: Configuring the Defenses

To configure the defense mode of the victim model, modify `./config-jsons/defense_config.json`. Instructions are provided in the [CIFAR-10 section](#step-1-configuring-the-defenses) above.

#### Configuring RFD

**For ResNet-50:**

```json
"rfd": {
    "sigma_noise": 3.0,
    "target_layers": [4]
}
```

### Step 2: Configuring the Attacks

The steps are similar to the CIFAR-10 case. Attack config files are as follows:

**For ImageNet in L∞:**
```
./config-jsons/imagenet_bandit_linf_config.json
./config-jsons/imagenet_nes_linf_config.json
./config-jsons/imagenet_sign_linf_config.json
./config-jsons/imagenet_square_linf_config.json
./config-jsons/imagenet_zosignsgd_linf_config.json
```

**For ImageNet in L2:**
```
./config-jsons/imagenet_bandit_l2_config.json
./config-jsons/imagenet_nes_l2_config.json
./config-jsons/imagenet_square_l2_config.json
./config-jsons/imagenet_zosignsgd_l2_config.json
```

Each config file has a `modeln` attribute that must be set to:

```json
"modeln": "Resnet50"
```

### Step 3: Running the Attack

Instead of `attack_cifar10.py`, use `attack_imagenet.py`:

```bash
python attack_imagenet.py [path_to_attack_config]
```

## Adaptive Attack against AAA-Sine

We develop an adaptive attack against AAA-Sine to demonstrate its vulnerability. To run this adaptive attack, use Square Attack and modify `./config-jsons/adaptive.json` as follows:

```json
{
    "method": "switch_dir",
    "switch_dir": {
        "k": 10
    },
    "none": {},
    "verbose": true
}
```

Setting `method` to `"none"` disables the adaptive attack. `k` under `switch_dir` specifies the number of unsuccessful attack iterations before the attacker switches direction, as explained in Section 4 of the paper.


## Citation

To cite our paper, you can use the following BibTeX entry:

```bibtex
@article{
rls2026,
title={Random Logit Scaling: Defending Deep Neural Networks Against Black-Box Score-Based Adversarial Example Attacks},
author={Hamid Dashtbani and Mehdi Dousti Gandomani and AmirMahdi Sadeghzadeh},
journal={Transactions on Machine Learning Research},
year={2026},
url={https://openreview.net/forum?id=CXafPv4aAG}
}
```
