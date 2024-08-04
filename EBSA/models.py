import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
import transformers
import tensorflow as tf
from tensorflow.keras import layers
from transformers.modeling_tf_utils import TFModelInputType

# Logger ayarları
logger = logging.getLogger('absa.model')

class ABSClassifier(tf.keras.Model, ABC):
    @abstractmethod
    def call(
            self,
            input_ids: tf.Tensor,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            training: bool = False,
            **bert_kwargs
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        """
        Bu metot, alt sınıflarda yeniden tanımlanacak ve modelin ileri besleme (forward pass) işlemini gerçekleştirecek.
        """

def force_to_return_details(kwargs: dict):
    """
    Modelin dikkat (attention) ve gizli durumları (hidden states) döndürmesini sağlar.
    
    Args:
    kwargs (dict): Modelin argümanları.
    """
    condition = not kwargs.get('output_attentions', False) or \
                not kwargs.get('output_hidden_states', False)
    if condition:
        logger.info('Model dikkat ve gizli durumları döndürmelidir.')
    kwargs['output_attentions'] = True
    kwargs['output_hidden_states'] = True

class BertABSCConfig(transformers.BertConfig):
    """
    BERT tabanlı Aspect-Based Sentiment Classification (ABSC) için yapılandırma sınıfı.
    """

    def __init__(self, num_polarities: int = 3, **kwargs):
        force_to_return_details(kwargs)
        super().__init__(**kwargs)
        self.num_polarities = num_polarities

class BertABSClassifier(ABSClassifier, transformers.TFBertPreTrainedModel):
    """
    BERT tabanlı Aspect-Based Sentiment Classification (ABSC) sınıflandırıcı modeli.
    """

    def __init__(self, config: BertABSCConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = transformers.TFBertMainLayer(config, name="bert")
        initializer = transformers.modeling_tf_utils.get_initializer(config.initializer_range)
        self.dropout = layers.Dropout(config.hidden_dropout_prob)
        self.classifier = layers.Dense(
            config.num_polarities,
            kernel_initializer=initializer,
            name='classifier'
        )

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        """
        Modelin ileri besleme (forward pass) işlemini gerçekleştirir.

        Args:
        input_ids (Optional[TFModelInputType]): Girdi token ID'leri.
        attention_mask (Optional[Union[np.ndarray, tf.Tensor]]): Dikkat maskesi.
        token_type_ids (Optional[Union[np.ndarray, tf.Tensor]]): Token tipi ID'leri.
        position_ids (Optional[Union[np.ndarray, tf.Tensor]]): Pozisyon ID'leri.
        head_mask (Optional[Union[np.ndarray, tf.Tensor]]): Kafa maskesi.
        inputs_embeds (Optional[Union[np.ndarray, tf.Tensor]]): Girdi gömmeleri.
        output_attentions (Optional[bool]): Dikkat çıktılarını döndür.
        output_hidden_states (Optional[bool]): Gizli durum çıktılarını döndür.
        return_dict (Optional[bool]): Çıktıları dict olarak döndür.
        training (Optional[bool]): Eğitim modu.

        Returns:
        Tuple[tf.Tensor, Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]: 
        Logits, gizli durumlar ve dikkat çıktılarını içeren tuple.
        """
        # Girdi işlemleri
        inputs = transformers.modeling_tf_utils.input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        # BERT modelinden çıktılar
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        # Sınıflandırıcı katmanı üzerinden geçiş
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits, outputs.hidden_states, outputs.attentions
