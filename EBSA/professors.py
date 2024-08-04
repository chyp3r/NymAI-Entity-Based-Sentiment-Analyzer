from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .aux_models import ReferenceRecognizer, PatternRecognizer
from .data_types import TokenizedExample, PredictedExample, Output, Review, Sentiment

@dataclass
class _Professor(ABC):
    """
    Professor sınıfının soyut temel sınıfı. 'review' metodunun arayüzünü tanımlar.
    """

    @abstractmethod
    def review(self, example: TokenizedExample, output_batch: Output) -> PredictedExample:
        """
        Bu metod, bir örneği (TokenizedExample) ve model çıktısını (Output) alır
        ve bir tahminli örnek (PredictedExample) döner.
        """
        pass

@dataclass
class Professor(_Professor):
    """
    _Professor sınıfının somut uygulamasıdır. Bu sınıf, model çıktısını gözden geçirir
    ve duygu tahmini yapar.
    """
    reference_recognizer: ReferenceRecognizer = None
    pattern_recognizer: PatternRecognizer = None

    def review(self, example: TokenizedExample, output: Output) -> PredictedExample:
        """
        Model çıktısını (Output) kullanarak verilen örnek üzerinde (TokenizedExample) inceleme yapar.
        Duygu tahmini yapar ve ilgili bilgileri içeren bir tahminli örnek (PredictedExample) döner.
        """
        # Çıktıdan skorları al ve NumPy dizisine dönüştür
        scores = list(output.scores.numpy())
        # En yüksek skoru olan duygu etiketini belirle
        sentiment_id = np.argmax(scores).astype(int)
        sentiment = Sentiment(sentiment_id)

        # Referans tanıyıcı varsa, referans olup olmadığını belirle
        is_reference = self.reference_recognizer(example, output) \
            if self.reference_recognizer else None
        # Desen tanıyıcı varsa, desenleri tanı
        patterns = self.pattern_recognizer(example, output) \
            if self.pattern_recognizer and is_reference is not False else None
        # İnceleme oluştur
        review = Review(is_reference, patterns)

        # Eğer inceleme referans değilse, duyguyu nötr olarak ayarla
        if review.is_reference is False:
            sentiment = Sentiment.neutral
            scores = [0, 0, 0]

        # PredictedExample nesnesini oluştur ve döndür
        prediction = PredictedExample.from_example(
            example, sentiment=sentiment, scores=scores, review=review)
        return prediction
