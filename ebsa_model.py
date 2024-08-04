import EBSA as absa

def ebsa_sentiment(aspects, text):
    """
    Belirli bir metin ve yöneltilen özellikler (aspects) için ABSA (Aspect-Based Sentiment Analysis) modelini kullanarak duygu analizi yapar.
    
    Args:
        aspects (list): Metinde analiz edilmesi gereken özelliklerin (aspects) listesi. Her özellik, duygu analizinin yapılacağı bir yön veya konu olabilir.
        text (str): Analiz edilecek metin.
        
    Returns:
        list: ABSA modelinin analiz sonuçlarını içeren bir liste. Her bir öğe, metindeki bir özelliğe ilişkin duygu skorlarını içerir.
        dict: Hata durumunda boş bir sözlük döner.
    """
    try:
        # ABSA modelini yükle
        nlp = absa.load()
        
        # Metni ve özellikleri kullanarak duygu analizini yap
        sentiments = nlp(text, aspects=aspects)
        
        return sentiments
    
    except Exception as e:
        # Hata durumunda boş bir sözlük döndür
        print(f"Hata: {e}")  # Hata mesajını ekrana yazdır
        return {}
