import logging
from datetime import datetime

from datasets import Cifar10ZCA
from mean_teacher.model import Model
from mean_teacher import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run():
    data_seed = 0
    date = datetime.now()
    n_labeled = 4000


    result_dir = "{root}/{dataset}/{model}/{date:%Y-%m-%d_%H:%M:%S}/{seed}".format(
        root='results/final_eval',
        dataset='cifar10_{}'.format(n_labeled),
        model='mean_teacher',
        date=date,
        seed=data_seed
    )

    model = Model(result_dir=result_dir)
    model['flip_horizontally'] = True
    model['max_consistency_coefficient'] = 100.0 * n_labeled / 50000
    model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 25000
    model['training_length'] = 150000

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar10ZCA(data_seed, n_labeled)
    training_batches = minibatching.training_batches(cifar.training)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
