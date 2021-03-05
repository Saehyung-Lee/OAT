# Removing Undesirable Feature Contributions using Out-of-Distribution Data

This repository is the official implementation of "Removing Undesirable Feature Contributions using Out-of-Distribution Data", published as a conference paper at [ICLR 2021](https://openreview.net/forum?id=eIHYL6fpbkA). 

## Requirements

To install requirements:

```setup
conda env create -f requirements.yml
```

You can download the OOD dataset and pre-trained models on CIFAR-10 here:

- [OOD dataset](https://drive.google.com/file/d/13Nyw3b8lBfBTbVnUEw_yyFGW7x6rjWRD/view?usp=sharing)

- [OAT+PGD](https://drive.google.com/file/d/1uvUECJJi3ccgWilNFHqoplGQPiE-iMmF/view?usp=sharing)

You can also create your own OOD datasets using the work of [Carmon et al.](https://github.com/yaircarmon/semisup-adv)

## Training

To train the OAT model(s), run this command:

```train
python train.py --oat --alpha 1.0 --suffix <model_dir_name>
```

> Please specify the path to OOD dataset in the config.json file.

## Evaluation

To evaluate the models on CIFAR-10, run:

```eval
python eval.py <num_steps> <xent or cw> <model_dir>
```

## Results

Our model achieves the following performance on CIFAR-10:

| Model name         | Classification Accuracy | Robustness (PGD10)     |
| ------------------ |------------------------ | ---------------------- |
| WRN-34-10+PGD      |     87.48%              |              51.93%    |
| WRN-34-10+PGD+OAT  |     86.63%              |              58.89%    |
