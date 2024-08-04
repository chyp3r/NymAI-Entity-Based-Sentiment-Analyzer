from typing import Iterable
from typing import List

import tensorflow as tf

from ..pipelines import BertABSClassifier
from .callbacks import Callback
from .data_types import ClassifierTrainBatch
from . import routines


def train_classifier(
        model: BertABSClassifier,
        optimizer: tf.keras.optimizers.Optimizer,
        train_dataset: Iterable[ClassifierTrainBatch],
        epochs: int,
        test_dataset: Iterable[ClassifierTrainBatch] = None,
        callbacks: List[Callback] = None,
        strategy: tf.distribute.Strategy = tf.distribute.OneDeviceStrategy('CPU')
):
    """
    BERT tabanlı sınıflandırıcı modelini belirtilen eğitim verileri üzerinde eğitir.

    :param model: Eğitim için kullanılan BERT tabanlı sınıflandırıcı model.
    :param optimizer: Modelin ağırlıklarını güncellemek için kullanılan optimizatör.
    :param train_dataset: Eğitim verilerini içeren iterable.
    :param epochs: Eğitim için kaç epoch kullanılacağı.
    :param test_dataset: (Opsiyonel) Test verilerini içeren iterable.
    :param callbacks: (Opsiyonel) Eğitim sırasında çağrılacak geri çağırmalar.
    :param strategy: (Opsiyonel) Dağıtık eğitim stratejisi.
    """
    with strategy.scope():

        def train_step(*batch: List[tf.Tensor]):
            """
            Bir eğitim batch'i için ileri geçiş ve geri yayılım adımlarını uygular.

            :param batch: Eğitim verilerini içeren batch.
            :return: Eğitim kaybı ve model çıktıları.
            """
            token_ids, attention_mask, token_type_ids, target_labels = batch
            with tf.GradientTape() as tape:
                model_outputs = model.call(
                    token_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    training=True
                )
                logits, *details = model_outputs
                loss_value = classifier_loss(target_labels, logits)

            variables = model.bert.trainable_variables \
                        + model.classifier.trainable_variables
            grads = tape.gradient(loss_value, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return [loss_value, *model_outputs]

        def test_step(*batch: List[tf.Tensor]):
            """
            Bir test batch'i için ileri geçiş adımını uygular.

            :param batch: Test verilerini içeren batch.
            :return: Test kaybı ve model çıktıları.
            """
            token_ids, attention_mask, token_type_ids, target_labels = batch
            model_outputs = model.call(
                token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits, *details = model_outputs
            loss_value = classifier_loss(target_labels, logits)
            return [loss_value, *model_outputs]

    routines.train(
        strategy=strategy,
        train_step=train_step,
        train_dataset=train_dataset,
        test_step=test_step,
        test_dataset=test_dataset,
        epochs=epochs,
        callbacks=callbacks
    )


def classifier_loss(labels, logits) -> tf.Tensor:
    """
    Sınıflandırıcı kaybını hesaplar.

    :param labels: Gerçek etiketler.
    :param logits: Model tarafından tahmin edilen logits.
    :return: Hesaplanan kayıp değeri.
    """
    softmax = tf.nn.softmax_cross_entropy_with_logits
    return softmax(labels, logits, axis=-1, name='Loss')
