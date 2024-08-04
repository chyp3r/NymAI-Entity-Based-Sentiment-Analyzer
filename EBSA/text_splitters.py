from typing import Callable, List
import spacy

def sentencizer(name: str = 'en_core_web_sm') -> Callable[[str], List[str]]:
    """
    Verilen Spacy model adını kullanarak bir metni cümlelere ayıran bir fonksiyon döndürür.
    
    Args:
        name (str): Yüklenecek Spacy modelinin adı. Varsayılan olarak 'en_core_web_sm' kullanılır.

    Returns:
        Callable[[str], List[str]]: Metni cümlelere bölen bir fonksiyon.
    """
    nlp = spacy.load(name)  # Spacy modelini yükler

    def wrapper(text: str) -> List[str]:
        """
        Verilen metni cümlelere böler.

        Args:
            text (str): Cümlelere ayrılacak metin.

        Returns:
            List[str]: Cümlelere ayrılmış metinlerin listesi.
        """
        doc = nlp(text)  # Metni işleyerek Spacy dokümanına dönüştürür
        sentences = [str(sent).strip() for sent in doc.sents]  # Cümleleri listeye dönüştürür ve beyaz boşlukları temizler
        return sentences

    return wrapper
