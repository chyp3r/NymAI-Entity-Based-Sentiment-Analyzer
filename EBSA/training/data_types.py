from abc import ABC
from dataclasses import dataclass
import tensorflow as tf


class TrainBatch(ABC):
    """
    Eğitim verileri için temel bir sınıf.
    Bu sınıf soyut bir sınıftır ve eğitim verilerini temsil eden özelliklere sahip alt sınıflar tarafından genişletilmelidir.
    """


@dataclass(frozen=True)
class ClassifierTrainBatch(TrainBatch):
    """
    Sınıflandırıcı modeli için eğitim batch'ini temsil eden veri sınıfı.

    :param token_ids: Modelin giriş token'larını temsil eden tensör.
    :param attention_mask: Modelin hangi token'lara dikkat etmesi gerektiğini belirten mask.
    :param token_type_ids: Token türlerini temsil eden tensör.
    :param target_labels: Modelin tahmin etmesi gereken gerçek etiketler.
    """
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor
    target_labels: tf.Tensor
