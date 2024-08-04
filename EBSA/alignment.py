from functools import partial
from typing import List, Tuple

import tensorflow as tf
import transformers
import numpy as np

from .data_types import TokenizedExample

def tokenize(
        tokenizer: transformers.BertTokenizer,
        text: str,
        aspect: str
) -> TokenizedExample:
    """
    Metin ve aspekt (özellik) için verilen bir tokenizer kullanarak tokenizasyon yapar.
    
    Args:
    tokenizer (transformers.BertTokenizer): BERT tokenizer.
    text (str): Tokenize edilecek metin.
    aspect (str): Metinle ilişkilendirilecek aspekt (özellik).

    Returns:
    TokenizedExample: Tokenize edilmiş örneği içeren bir nesne.
    """
    # Temel ve wordpiece tokenleştiricileri al
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    # Metni tokenleştir
    text_tokens = basic_tokenizer.tokenize(text)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]

    # Aspekt varsa tokenleştir, yoksa None
    aspect_tokens = basic_tokenizer.tokenize(aspect) if aspect else None

    # Aspekt varsa token dizisini birleştir, yoksa sadece metin tokenlerini kullan
    tokens = cls + text_tokens + sep + aspect_tokens + sep if aspect else cls + text_tokens + sep

    # Subtokenları al
    aspect_subtokens = get_subtokens(wordpiece_tokenizer, aspect_tokens)
    text_subtokens = get_subtokens(wordpiece_tokenizer, text_tokens)

    # Hizalamayı yap
    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, tokens)

    # TokenizedExample nesnesi oluştur ve döndür
    example = TokenizedExample(
        text=text,
        text_tokens=text_tokens,
        text_subtokens=text_subtokens,
        aspect=aspect,
        aspect_tokens=aspect_tokens,
        aspect_subtokens=aspect_subtokens,
        tokens=tokens,
        subtokens=sub_tokens,
        alignment=alignment
    )
    return example


def get_subtokens(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> List[str]:
    """
    Verilen tokenler için subtokens (alt tokenler) oluşturur.

    Args:
    tokenizer (transformers.WordpieceTokenizer): Wordpiece tokenizer.
    tokens (List[str]): Token listesi.

    Returns:
    List[str]: Alt tokenlerin listesi.
    """
    split = tokenizer.tokenize
    return [sub_token for token in tokens for sub_token in split(token)]


def make_alignment(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> Tuple[List[str], List[List[int]]]:
    """
    Tokenler ve subtokenlar arasında hizalama yapar.

    Args:
    tokenizer (transformers.WordpieceTokenizer): Wordpiece tokenizer.
    tokens (List[str]): Token listesi.

    Returns:
    Tuple[List[str], List[List[int]]]: Subtokenlar ve hizalama indekslerinin listesi.
    """
    i = 0
    sub_tokens = []
    alignment = []
    for token in tokens:
        indices = []
        word_pieces = tokenizer.tokenize(token)
        for sub_token in word_pieces:
            indices.append(i)
            sub_tokens.append(sub_token)
            i += 1

        alignment.append(indices)
    return sub_tokens, alignment


def merge_tensor(tensor: tf.Tensor, alignment: List[List[int]]) -> tf.Tensor:
    """
    Tensor verisini hizalamaya göre birleştirir.

    Args:
    tensor (tf.Tensor): Giriş tensörü.
    alignment (List[List[int]]): Hizalama indeksleri.

    Returns:
    tf.Tensor: Birleştirilmiş tensör.
    """
    def aggregate(a, fun):
        n = len(alignment)
        new = np.zeros(n)
        for i in range(n):
            new[i] = fun(a[alignment[i]])
        return new
    
    x = tensor.numpy()
    # İlgili elementleri ortalama alarak birleştir
    attention_to = partial(aggregate, fun=np.mean)
    x = np.apply_along_axis(attention_to, 2, x)
    # İlgili elementleri toplayarak birleştir
    attention_from = partial(aggregate, fun=np.sum)
    x = np.apply_along_axis(attention_from, 3, x)
    x = tf.convert_to_tensor(x)
    return x
