from enum import IntEnum
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import tensorflow as tf


class Sentiment(IntEnum):
    """
    Duygu durumu sınıflarını tanımlar.
    """
    neutral = 0
    negative = 1
    positive = 2


@dataclass(frozen=True)
class Example:
    """
    Temel örnek sınıfı.
    """
    text: str
    aspect: str


@dataclass(frozen=True)
class LabeledExample(Example):
    """
    Etiketlenmiş örnek sınıfı.
    """
    sentiment: Sentiment


@dataclass(frozen=True)
class TokenizedExample:
    """
    Tokenize edilmiş örnek sınıfı.
    """
    text: str
    text_tokens: List[str]
    text_subtokens: List[str]
    aspect: str
    aspect_tokens: List[str]
    aspect_subtokens: List[str]
    tokens: List[str]
    subtokens: List[str]
    alignment: List[List[int]]


@dataclass(frozen=True)
class Pattern:
    """
    Desen sınıfı.
    """
    importance: float
    tokens: List[str]
    weights: List[float]


@dataclass(frozen=True)
class Review:
    """
    İnceleme sınıfı.
    """
    is_reference: bool = None
    patterns: List[Pattern] = None


@dataclass(frozen=True)
class PredictedExample(TokenizedExample, LabeledExample):
    """
    Tahmin edilen örnek sınıfı.
    """
    scores: List[float]
    review: Review = None

    @classmethod
    def from_example(cls, example: TokenizedExample, **kwargs):
        """
        Tokenize edilmiş örnekten tahmin edilen örnek oluşturur.
        
        Args:
        example (TokenizedExample): Tokenize edilmiş örnek.
        **kwargs: Ek argümanlar.

        Returns:
        PredictedExample: Tahmin edilen örnek.
        """
        return cls(**asdict(example), **kwargs)


@dataclass(frozen=True)
class SubTask:
    """
    Alt görev sınıfı.
    """
    text: str
    aspect: str
    examples: List[Example]

    def __iter__(self) -> Iterable[Example]:
        return iter(self.examples)


@dataclass(frozen=True)
class CompletedSubTask(SubTask):
    """
    Tamamlanmış alt görev sınıfı.
    """
    examples: List[PredictedExample]
    sentiment: Sentiment
    scores: List[float]


@dataclass(frozen=True)
class Task:
    """
    Görev sınıfı.
    """
    text: str
    aspects: List[str]
    subtasks: Dict[str, SubTask]

    @property
    def indices(self) -> List[Tuple[int, int]]:
        """
        Alt görevlerin başlangıç ve bitiş indekslerini döndürür.
        
        Returns:
        List[Tuple[int, int]]: Başlangıç ve bitiş indeksleri.
        """
        indices = []
        start, end = 0, 0
        for subtask in self:
            length = len(list(subtask))
            end += length
            indices.append((start, end))
            start += length
        return indices

    @property
    def examples(self) -> List[Example]:
        """
        Tüm alt görevlerdeki örnekleri döndürür.
        
        Returns:
        List[Example]: Tüm örnekler.
        """
        return [example for subtask in self for example in subtask]

    def __getitem__(self, aspect: str) -> SubTask:
        """
        Belirtilen aspekt için alt görevi döndürür.
        
        Args:
        aspect (str): Aspekt.

        Returns:
        SubTask: Alt görev.
        """
        return self.subtasks[aspect]

    def __iter__(self) -> Iterable[SubTask]:
        """
        Alt görevleri iteratif olarak döndürür.
        
        Returns:
        Iterable[SubTask]: Alt görevler.
        """
        return (self[aspect] for aspect in self.aspects)


@dataclass(frozen=True)
class CompletedTask(Task):
    """
    Tamamlanmış görev sınıfı.
    """
    subtasks: Dict[str, CompletedSubTask]


@dataclass(frozen=True)
class InputBatch:
    """
    Girdi batch sınıfı.
    """
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor


@dataclass(frozen=True)
class Output:
    """
    Model çıktısı sınıfı.
    """
    scores: tf.Tensor  
    hidden_states: tf.Tensor
    attentions: tf.Tensor
    attention_grads: tf.Tensor 


@dataclass(frozen=True)
class OutputBatch:
    """
    Model batch çıktısı sınıfı.
    """
    scores: tf.Tensor 
    hidden_states: tf.Tensor 
    attentions: tf.Tensor 
    attention_grads: tf.Tensor 

    def __getitem__(self, i: int) -> Output:
        """
        Belirtilen indeksteki çıktıyı döndürür.
        
        Args:
        i (int): İndeks.

        Returns:
        Output: Çıktı.
        """
        return Output(
            self.scores[i],
            self.hidden_states[i],
            self.attentions[i],
            self.attention_grads[i]
        )

    def __iter__(self) -> Iterable[Output]:
        """
        Çıktıları iteratif olarak döndürür.
        
        Returns:
        Iterable[Output]: Çıktılar.
        """
        num_examples, classes = self.scores.shape
        return (self[i] for i in range(num_examples))
