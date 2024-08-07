from utils_text import translate_to_en, space_handler, clean_text, output_formater
from ner import special_cases, find_company_names, change_tr_ner
from ebsa_model import ebsa_sentiment

class MainModel:
    """
    Metin işleme ve duygu analizi yapan ana model sınıfı.
    
    Özellikler:
        company_set (set): Metinden çıkarılan ve işlenen şirket adlarını tutan küme.
        output (dict): Duygu analizi sonuçlarını ve diğer bilgileri içeren çıktı.
    """
    
    def __init__(self):
        """
        Sınıfın başlatıcısı. Başlangıçta boş bir şirket kümesi ve boş bir çıktı oluşturur.
        """
        self.company_set = set()
        self.output = {}

    def execute_model(self, text):
        """
        Verilen metin üzerinde bir dizi işlem gerçekleştirir:
        1. Metni İngilizceye çevirir.
        2. Boşlukları ve özel karakterleri düzenler.
        3. Temizler ve şirket isimlerini ayıklar.
        4. Şirket isimlerini İngilizce ve Türkçe olarak işler.
        5. ABSA modelini kullanarak duygu analizi yapar.
        6. Sonuçları formatlar ve saklar.
        
        Args:
            text (str): İşlenecek metin.
        """
        # Metni İngilizceye çevir
        translated_text = translate_to_en(text)
        
        # Metindeki fazla boşlukları ve özel karakterleri düzenle
        translated_text = space_handler(translated_text)
        
        # Metni temizle
        cleaned_text = clean_text(translated_text)
        
        # Özel durumları işleyerek şirket isimlerini ayıkla
        special_cases(cleaned_text, self.company_set)
        
        # Şirket isimlerini bul ve işleyerek kümesine ekle
        find_company_names(cleaned_text, self.company_set)
        
        # Şirket isimlerini Türkçe'den İngilizce'ye çevir ve metindeki en yakın eşleşenleri bul
        ner_results = change_tr_ner(text, self.company_set)
        
        # ABSA modelini kullanarak duygu analizi yap
        sentiments = ebsa_sentiment(ner_results.keys(), cleaned_text)
        
        # Sonuçları formatla ve sakla
        self.output = output_formater(sentiments, ner_results)

    def take_outputs(self):
        """
        Analiz sonuçlarını döndürür ve ekrana yazdırır.
        
        Returns:
            dict: Analiz sonuçlarını içeren sözlük.
        """
        print(self.output)
        return self.output
