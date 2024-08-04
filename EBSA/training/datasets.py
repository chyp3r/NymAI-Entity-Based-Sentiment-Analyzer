from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List

import numpy as np
import tensorflow as tf
import transformers

from . import ClassifierTrainBatch
from .data_types import TrainBatch
from ..data_types import LabeledExample


class Dataset(ABC):
    """
    Veri kümesi için soyut bir temel sınıf. 
    Eğitim ve test verilerini iterasyon yapabilen veri kümesi sınıfları bu sınıftan türemelidir.
    """

    @abstractmethod
    def __iter__(self) -> Iterable[TrainBatch]:
        """
        Veri kümesini iterasyon yaparak döndürür.
        Bu metod, veri kümesinin her bir batch'ini iterasyon yaparak döndürmelidir.
        """
        pass

    @abstractmethod
    def preprocess_batch(self, batch_examples: List[Any]) -> TrainBatch:
        """
        Verilen örnekleri işleyip bir batch oluşturur.

        :param batch_examples: İşlenecek örneklerin listesi.
        :return: İşlenmiş batch.
        """
        pass


class InMemoryDataset(Dataset, ABC):
    """
    Bellekte saklanan veri kümesi sınıfı. 
    Veriler bellekte tutulur ve batch'ler oluşturulur.
    """

    examples: List
    batch_size: int

    def __iter__(self) -> Iterator[TrainBatch]:
        """
        Veri kümesini iterasyon yaparak döndürür. 
        Veriler rastgele sıralanır ve batch'ler halinde döndürülür.
        """
        order = np.random.permutation(len(self.examples))
        batch_examples = []
        for index in order:
            example = self.examples[index]
            batch_examples.append(example)
            if len(batch_examples) == self.batch_size:
                batch = self.preprocess_batch(batch_examples)
                yield batch
                batch_examples = []


class StreamDataset(Dataset, ABC):
    """
    Akış tabanlı veri kümesi sınıfı.
    Veriler bir akıştan gelir ve batch'ler oluşturulur.
    """

    batch_size: int

    def __iter__(self) -> Iterator[TrainBatch]:
        """
        Veri akışını iterasyon yaparak döndürür.
        Veriler akıştan alınır ve batch'ler halinde döndürülür.
        """
        examples = self.examples_generator()
        batch_examples = []
        for example in examples:
            batch_examples.append(example)
            if len(batch_examples) == self.batch_size:
                batch_input = self.preprocess_batch(batch_examples)
                yield batch_input
                batch_examples = []

    @abstractmethod
    def examples_generator(self) -> Iterable[Any]:
        """
        Verilerin üretildiği örnek üretici metod.
        Bu metod, verileri iterasyon yaparak döndürmelidir.
        """
        pass


@dataclass(frozen=True)
class ClassifierDataset(InMemoryDataset):
    """
    Sınıflandırıcı için veri kümesi sınıfı. 
    Eğitim verilerini işler ve batch'ler oluşturur.
    """

    examples: List[LabeledExample]
    batch_size: int
    tokenizer: transformers.PreTrainedTokenizer
    num_polarities: int = 3

    def preprocess_batch(
            self, batch_examples: List[LabeledExample]
    ) -> ClassifierTrainBatch:
        """
        Verilen örnekleri işleyip bir sınıflandırıcı batch'i oluşturur.

        :param batch_examples: İşlenecek etiketli örneklerin listesi.
        :return: İşlenmiş sınıflandırıcı batch'i.
        """
        pairs = [(e.text, e.aspect) for e in batch_examples]
        encoded = self.tokenizer.batch_encode_plus(
            pairs,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        sentiments = [e.sentiment for e in batch_examples]
        target_labels = tf.one_hot(sentiments, depth=self.num_polarities)
        train_batch = ClassifierTrainBatch(
            input_ids,
            attention_mask,
            token_type_ids,
            target_labels
        )
        return train_batch

    @classmethod
    def from_iterable(cls, examples: Iterable[LabeledExample], *args, **kwargs):
        """
        Bir iteratörden ClassifierDataset örneği oluşturur.

        :param examples: LabeledExample örneklerinin iteratörü.
        :return: ClassifierDataset örneği.
        """
        examples = list(examples)
        return cls(examples, *args, **kwargs)
