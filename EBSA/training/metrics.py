import tensorflow as tf


class ConfusionMatrix(tf.metrics.Metric):
    """
    Sinir ağı modellerinin performansını değerlendirmek için kullanılan bir karışıklık matrisini temsil eden sınıftır.
    Karışıklık matrisi, gerçek etiketler ile tahmin edilen etiketler arasındaki ilişkiyi gösterir.

    Attributes:
    - num_classes (int): Toplam sınıf sayısı.
    - data (tf.Variable): Karışıklık matrisini saklayan değişken.
    """

    def __init__(self, num_classes: int):
        """
        Karışıklık matrisini başlatır.

        Args:
        - num_classes (int): Toplam sınıf sayısı.
        """
        super().__init__()
        self.num_classes = num_classes
        self.data = self.add_weight(name='confusion-matrix',
                                    shape=[num_classes, num_classes],
                                    initializer='zeros',
                                    dtype=tf.dtypes.int32)

    def update_state(self, y_true, y_pred):
        """
        Modelin tahminleri ve gerçek etiketlerle karışıklık matrisini günceller.

        Args:
        - y_true (tf.Tensor): Gerçek etiketler.
        - y_pred (tf.Tensor): Modelin tahminleri.
        """
        batch = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.dtypes.int32
        )
        self.data.assign_add(batch)

    def result(self):
        """
        Güncellenmiş karışıklık matrisini döndürür.
        """
        return self.data
