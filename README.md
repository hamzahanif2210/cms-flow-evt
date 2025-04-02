# Official repository for "Conditional Deep Generative Models for Simultaneous Simulation and Reconstruction of Entire Events" paper

This repository contains the code for the paper "Conditional Deep Generative Models for Simultaneous Simulation and Reconstruction of Entire Events", [arXiv:2503.19981](https://arxiv.org/abs/2503.19981).

The dataset used in the paper can be found [here](https://zenodo.org/records/15083495).

## Requirements
The list of packages required to train/evaluate model is found at `requirements.txt` file. All studies were done with `Python 3.11.9`.

## Training

The training script is provided in the `train.py` file. The script can be run as follows:

```bash
python train.py -c <path_to_config_file> --gpus 0
```

## Evaluation

The evaluation script is provided in the `eval.py` file. The script can be run as follows:

```bash
python eval.py \
-c <path_to_part_net_config> -p <path_to_part_net_checkpoint> \
-ce <path_to_evt_net_config> -pe <path_to_evt_net_checkpoint> \
--test_path <path_to_test_file> -ne <number_of_events> -bs <batch_size> \
-n <num_steps> [--prefix <prefix>]
```

The pre-trained model used in the paper can be found in the `trained_models` folder.

Example:
```bash
python eval.py \
-c trained_models/part.yaml -p trained_models/cms_part_epoch=66.ckpt \
-ce trained_models/evt.yaml -pe trained_models/cms_evt_epoch=37.ckpt \
--test_path h4lep_test_cms100k.root -ne 10000 -bs 2000 \
-n 40 --prefix h4lep
```