# Antimicrobial peptide prediction with transformer and imbalanced learning

This is a method for integrating transformer and imbalanced multi-label learning to identify the functional activities of targets for antimicrobial peptides

## Requirements

We have already integrate the environment in `env.yaml`. execute `conda create -f env.yaml` to install packages required in a new created `DEEPSEQENV` conda env.

The environment is based on `pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0`. You may refer to [here](https://pytorch.org/get-started/) to custom the pytorch environment for your own machine. 

Enter the enviornment with `conda activate DEEPSEQENV` before further executions.

## Training establishment

Execute `train.py` to start a training process for establishing AMP prediction model with different missions (make sure the pretrained tape model is downloaded before training). For example, to train a model for AMP identification with proper hyper-parameters, you can run:
```
python train.py  --cuda --seed 810 --task "AMP" --lr 0.03 --ckpt-iter 60 -e 256 -b 64 -d "trial-amp"
```
to train a model for functional activity prediction, for instance, simply run:
```
python train.py  --cuda --task "mtl" -d "trial-mtl"
```
For more information about parser arguments, please refer to the `train.py`.

## Evaluation

To evaluate the model, you can use `evaluate.py` with parsing the well-trained result path. For example, for the mentioned AMP identification trial:
```
python evaluate.py --path "./trial-amp" --cuda True
```