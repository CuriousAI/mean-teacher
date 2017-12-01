import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run():
    data_seed = 0
    n_labeled = 500
    n_extra_unlabeled = 0

    model = Model(RunContext(__file__, 0))
    model['rampdown_length'] = 0
    model['training_length'] = 180000

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    svhn = SVHN(data_seed, n_labeled, n_extra_unlabeled)
    training_batches = minibatching.training_batches(svhn.training, n_labeled_per_batch=1)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
