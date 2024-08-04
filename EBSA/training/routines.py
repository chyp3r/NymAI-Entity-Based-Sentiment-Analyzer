import dataclasses
import logging
from typing import Callable, Iterable, List

import numpy as np
import tensorflow as tf

from .callbacks import Callback, CallbackList
from .data_types import TrainBatch
from .errors import StopTraining

logger = logging.getLogger('absa.routines')


def train(strategy: tf.distribute.Strategy,
          train_step: Callable,
          train_dataset: Iterable[TrainBatch],
          test_step: Callable = None,
          test_dataset: Iterable[TrainBatch] = None,
          epochs: int = 10,
          callbacks: List[Callback] = None):
    """
    Modeli eğitmek için genel bir eğitim döngüsü sağlar.

    Args:
    - strategy (tf.distribute.Strategy): Dağıtılmış eğitim stratejisi.
    - train_step (Callable): Eğitim adımını gerçekleştiren fonksiyon.
    - train_dataset (Iterable[TrainBatch]): Eğitim verisi.
    - test_step (Callable, optional): Test adımını gerçekleştiren fonksiyon.
    - test_dataset (Iterable[TrainBatch], optional): Test verisi.
    - epochs (int, optional): Eğitim dönemi sayısı.
    - callbacks (List[Callback], optional): Eğitim sırasında çağrılacak geri çağırmalar.
    """
    callbacks = CallbackList(callbacks if callbacks else [])
    try:
        for epoch in np.arange(1, epochs+1):
            callbacks.on_epoch_begin(epoch)  # Dönem başlangıç geri çağırmalarını yap
            train_loop(train_step, train_dataset, callbacks, strategy)  # Eğitim döngüsünü çalıştır
            if test_step and test_dataset:
                test_loop(test_step, test_dataset, callbacks, strategy)  # Test döngüsünü çalıştır
            callbacks.on_epoch_end(epoch)  # Dönem bitiş geri çağırmalarını yap
    except StopTraining:
        logger.info('Eğitim rutini durduruldu.')


def train_loop(train_step: Callable,
               dataset: Iterable[TrainBatch],
               callbacks: Callback,
               strategy: tf.distribute.Strategy):
    """
    Eğitim adımını her bir eğitim verisi için çalıştıran döngüyü sağlar.

    Args:
    - train_step (Callable): Eğitim adımını gerçekleştiren fonksiyon.
    - dataset (Iterable[TrainBatch]): Eğitim verisi.
    - callbacks (Callback): Eğitim sırasında geri çağırmalar.
    - strategy (tf.distribute.Strategy): Dağıtılmış eğitim stratejisi.
    """
    step = wrap_step_into_strategy(train_step, strategy)
    for i, batch in enumerate(dataset):
        tf_batch = dataclasses.astuple(batch)  # Veriyi tuple'a dönüştür
        train_step_outputs = step(tf_batch)  # Eğitim adımını uygula
        callbacks.on_train_batch_end(i, batch, *train_step_outputs)  # Eğitim batch sonu geri çağırmalarını yap


def test_loop(test_step: Callable,
              dataset: Iterable[TrainBatch],
              callbacks: Callback,
              strategy: tf.distribute.Strategy):
    """
    Test adımını her bir test verisi için çalıştıran döngüyü sağlar.

    Args:
    - test_step (Callable): Test adımını gerçekleştiren fonksiyon.
    - dataset (Iterable[TrainBatch]): Test verisi.
    - callbacks (Callback): Test sırasında geri çağırmalar.
    - strategy (tf.distribute.Strategy): Dağıtılmış eğitim stratejisi.
    """
    step = wrap_step_into_strategy(test_step, strategy)
    for i, batch in enumerate(dataset):
        tf_batch = dataclasses.astuple(batch)  # Veriyi tuple'a dönüştür
        test_step_outputs = step(tf_batch)  # Test adımını uygula
        callbacks.on_test_batch_end(i, batch, *test_step_outputs)  # Test batch sonu geri çağırmalarını yap


def wrap_step_into_strategy(step: Callable, strategy: tf.distribute.Strategy):
    """
    Adım fonksiyonunu dağıtılmış stratejiye uygun şekilde sarar.

    Args:
    - step (Callable): Eğitim veya test adımını gerçekleştiren fonksiyon.
    - strategy (tf.distribute.Strategy): Dağıtılmış eğitim stratejisi.

    Returns:
    - Callable: Dağıtılmış stratejiye uygun olarak adım fonksiyonunu saran fonksiyon.
    """
    def one_device(batch):
        return strategy.run(step, args=batch)  # Tek bir cihaz için çalıştır

    def distributed(batch):
        dataset = tf.data.Dataset.from_tensors(batch)  # Tensorları veri kümesine dönüştür
        dist_batch, = strategy.experimental_distribute_dataset(dataset)  # Dağıtılmış veri kümesi oluştur
        per_replica_outputs = strategy.run(step, args=dist_batch)  # Dağıtılmış adım fonksiyonunu çalıştır
        with tf.device('CPU'):
            return [tf.concat(per_replica_output.values, axis=0)
                    for per_replica_output in per_replica_outputs]  # Çıktıları birleştir

    is_distributed = isinstance(strategy, tf.distribute.MirroredStrategy)
    return distributed if is_distributed else one_device
