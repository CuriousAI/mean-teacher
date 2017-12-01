# Mean teacher

This is the TensorFlow source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install tensorflow==1.2.1 numpy scipy
./download_svhn.py
```

To train the model, run:

* `python train_svhn.py` to train on SVHN using 500 labels
* `python train_cifar10.py` to train on CIFAR-10 using 4000 labels

To reproduce the results of the poster above, run:

* `python train_svhn_final_eval.py` to reproduce the SVHN results
* `python train_cifar10_final_eval.py` to reproduce the CIFAR-10 results
