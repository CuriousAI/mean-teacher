# Mean teacher

This is the source code for the ICLR 2017 workshop poster "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results".

- [ICLR 2017 workshop poster](ICLR_2017_poster.pdf)

For a longer description, see the papers below. They are currently slighty out-of-date and report worse results than the code in this repository. The main difference is that the runs in the papers used batch normalization and this repository uses weight normalization. The results in the poster above were achieved with the code in this repository.

- [ICLR 2017 workshop paper](https://openreview.net/pdf?id=ry8u21rtl)
- [A longer version of the paper in Arxiv](https://arxiv.org/abs/1703.01780)

## Usage

The code runs on Python 3. Install the dependencies and download the SVHN dataset with the commands below:

```
pip install tensorflow numpy scipy
./download_svhn.py
```

To train the model run

```
python train_svhn.py
```

To reproduce the results of the poster above, run

```
python train_svhn_all_final_eval.py
```
