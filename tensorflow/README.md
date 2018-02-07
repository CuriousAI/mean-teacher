# Mean Teacher using TensorFlow

This is the TensorFlow source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install tensorflow==1.2.1 numpy scipy pandas
./prepare_data.sh
```

Note that TensorFlow versions >= 1.3 have [a bug](https://github.com/tensorflow/tensorflow/issues/13351) that causes the code to hang up in the beginning. If you want to provide a workaround, [pull requests are welcome](../../../issues/1).

To train the model, run:

* `python train_svhn.py` to train on SVHN using 500 labels
* `python train_cifar10.py` to train on CIFAR-10 using 4000 labels

To reproduce the experiments in the paper run: `python -m experiments.cifar10_final_eval` or similar.
See the experiments directory for all the experiments.
