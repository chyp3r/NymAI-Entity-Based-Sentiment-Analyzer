from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from transformers import PretrainedConfig

from .data_types import Pattern, TokenizedExample, Output
from . import alignment


class ReferenceRecognizer(ABC):
    """
    Referans tanıma için soyut sınıf.
    """

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> bool:
        pass


class PatternRecognizer(ABC):
    """
    Pattern (desen) tanıma için soyut sınıf.
    """

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        """
        Bir örnek ve model çıktısına göre desenleri döndürür.
        """


@dataclass
class BasicReferenceRecognizer(ReferenceRecognizer, PretrainedConfig):
    weights: Tuple[float, float]
    model_type: str = 'reference_recognizer'

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> bool:
        """
        Referans tanıma işlemi yapar.
        
        Args:
        example (TokenizedExample): Tokenize edilmiş örnek.
        output (Output): Model çıktısı.

        Returns:
        bool: Referans olup olmadığını belirten değer.
        """
        β_0, β_1 = self.weights
        n = len(example.subtokens)
        hidden_states = output.hidden_states[:, :n, :]
        text_mask, aspect_mask = self.text_aspect_subtoken_masks(example)
        similarity = self.transform(hidden_states, text_mask, aspect_mask)
        is_reference = β_0 + β_1 * similarity > 0
        return bool(is_reference)   

    @staticmethod
    def transform(
            hidden_states: tf.Tensor,
            text_mask: List[bool],
            aspect_mask: List[bool]
    ) -> float:
        """
        Gizli durumları benzerlik skoruna dönüştürür.
        
        Args:
        hidden_states (tf.Tensor): Gizli durumları içeren tensör.
        text_mask (List[bool]): Metin maskesi.
        aspect_mask (List[bool]): Aspekt maskesi.

        Returns:
        float: Benzerlik skoru.
        """
        hidden_states = hidden_states.numpy()
        h = hidden_states[0, ...]  
        h_t = h[text_mask, :].mean(axis=0)
        h_a = h[aspect_mask, :].mean(axis=0)

        # L2 normuna göre normalizasyon
        h_t /= np.linalg.norm(h_t, ord=2)
        h_a /= np.linalg.norm(h_a, ord=2)

        similarity = h_t @ h_a
        return similarity

    @staticmethod
    def text_aspect_subtoken_masks(
            example: TokenizedExample
    ) -> Tuple[List[bool], List[bool]]:
        """
        Metin ve aspekt için subtoken maskeleri oluşturur.
        
        Args:
        example (TokenizedExample): Tokenize edilmiş örnek.

        Returns:
        Tuple[List[bool], List[bool]]: Metin ve aspekt maskeleri.
        """
        text = np.zeros(len(example.subtokens)).astype(bool)
        text[1:len(example.text_subtokens)+1] = True
        aspect = np.zeros(len(example.subtokens)).astype(bool)
        aspect[-(len(example.aspect_subtokens) + 1):-1] = True
        return text.tolist(), aspect.tolist()


@dataclass
class BasicPatternRecognizer(PatternRecognizer):
    max_patterns: int = 5
    is_scaled: bool = True
    is_rounded: bool = True
    round_decimals: int = 2

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        """
        Desen tanıma işlemi yapar.
        
        Args:
        example (TokenizedExample): Tokenize edilmiş örnek.
        output (Output): Model çıktısı.

        Returns:
        List[Pattern]: Bulunan desenlerin listesi.
        """
        text_mask = self.text_tokens_mask(example)
        w, pattern_vectors = self.transform(output, text_mask, example.alignment)
        patterns = self.build_patterns(w, example.text_tokens, pattern_vectors)
        return patterns

    def transform(
            self,
            output: Output,
            text_mask: List[bool],
            token_subtoken_alignment: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model çıktısını ve metin maskesini kullanarak desenleri dönüştürür.
        
        Args:
        output (Output): Model çıktısı.
        text_mask (List[bool]): Metin maskesi.
        token_subtoken_alignment (List[List[int]]): Token-subtoken hizalaması.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Ağırlıklar ve desen vektörleri.
        """
        x = output.attentions * tf.abs(output.attention_grads)
        x = tf.reduce_sum(x, axis=[0, 1], keepdims=True)
        x = alignment.merge_tensor(x, alignment=token_subtoken_alignment)
        x = x.numpy().squeeze(axis=(0, 1))

        w = x[0, text_mask]
        w /= np.max(w + 1e-9)

        patterns = x[text_mask, :][:, text_mask]
        max_values = np.max(patterns + 1e-9, axis=1)
        np.fill_diagonal(patterns, max_values)
        patterns /= max_values.reshape(-1, 1)

        if self.is_scaled:
            patterns *= w.reshape(-1, 1)
        if self.is_rounded:
            w = np.round(w, decimals=self.round_decimals)
            patterns = np.round(patterns, decimals=self.round_decimals)
        return w, patterns

    @staticmethod
    def text_tokens_mask(example: TokenizedExample) -> List[bool]:
        """
        Metin token maskesi oluşturur.
        
        Args:
        example (TokenizedExample): Tokenize edilmiş örnek.

        Returns:
        List[bool]: Metin token maskesi.
        """
        mask = np.zeros(len(example.tokens)).astype(bool)
        mask[1:len(example.text_tokens) + 1] = True
        return mask.tolist()

    def build_patterns(
            self,
            w: np.ndarray,
            tokens: List[str],
            pattern_vectors: np.ndarray
    ) -> List[Pattern]:
        """
        Ağırlıklar ve desen vektörlerini kullanarak desenleri oluşturur.
        
        Args:
        w (np.ndarray): Ağırlıklar.
        tokens (List[str]): Token listesi.
        pattern_vectors (np.ndarray): Desen vektörleri.

        Returns:
        List[Pattern]: Desenlerin listesi.
        """
        indices = np.argsort(w * -1)
        build = lambda i: Pattern(w[i], tokens, pattern_vectors[i, :].tolist())
        return [build(i) for i in indices[:self.max_patterns]]


def predict_key_set(patterns: List[Pattern], n: int) -> Set[int]:
    """
    Belirtilen desenlere göre önemli anahtar setini tahmin eder.
    
    Args:
    patterns (List[Pattern]): Desenlerin listesi.
    n (int): Anahtar sayısı.

    Returns:
    Set[int]: Anahtar seti.
    """
    weights = np.stack([p.weights for p in patterns]).sum(axis=0)
    decreasing = np.argsort(weights * -1)
    key_set = set(decreasing[:n])
    return key_set
