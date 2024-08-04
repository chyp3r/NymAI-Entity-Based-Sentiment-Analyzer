import spacy
from utils_text import translate_to_tr, translate_to_en
from utils_ner import create_distance_list, find_closest_word
from download_manager import ensure_en_core_web

# Gerekli dil modellerini indir
ensure_en_core_web("en_core_web_lg")
ensure_en_core_web("en_core_web_sm")

# SpaCy dil modellerini yükle
nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")

def special_cases(text, company_set):
    """
    Metindeki özel durumları işler, özellikle '@' işareti ile başlayan şirket adlarını toplar.
    
    Args:
        text (str): İşlenecek metin.
        company_set (set): Şirket adlarını tutacak küme.
    """
    text = text.split(" ")
    for i in text:
        if i.startswith("@"):
            company_set.add(i[1:])  # '@' işaretini kaldırıp şirket adını küme ekle

def find_company_names(text, company_set): 
    """
    Metindeki şirket isimlerini tespit eder ve İngilizceye çevirir.
    
    Args:
        text (str): Şirket isimlerini bulmak için işlenecek metin.
        company_set (set): Şirket adlarını içeren küme.
    """
    # SpaCy'nin küçük modelini kullanarak metni işle
    doc_sm = nlp_sm(text)
    
    # SpaCy'nin büyük modelini kullanarak metni işle
    doc_lg = nlp_lg(text)
    
    # Küçük model ve büyük modelden bulunan şirket adlarını al
    companies_sm = [ent.text.replace("@", "") for ent in doc_sm.ents if ent.label_ == "ORG"]
    companies_lg = [ent.text.replace("@", "") for ent in doc_lg.ents if ent.label_ == "ORG"]
    
    # Şirket adlarını İngilizceye çevir ve küme olarak ekle
    company_set |= set(map(lambda x: translate_to_en(x), set(map(lambda x: translate_to_tr(x), companies_sm + companies_lg))))

def change_tr_ner(text, company_set):
    """
    Şirket isimlerini metinde arar ve bulamazsa en yakın eşleşeni bulur.
    
    Args:
        text (str): Şirket isimlerini aramak için işlenecek metin.
        company_set (set): Şirket adlarını içeren küme.
        
    Returns:
        dict: Metinde bulunan şirket adlarını ve en yakın eşleşenlerini içeren sözlük.
    """
    final_entity_list = {}
    for i in company_set:
        if i.lower() not in text.lower():
            translated = translate_to_tr(i)
            if translated in text:
                final_entity_list[i] = translated
            else:
                # En yakın eşleşeni bulmak için metni parçalara ayır ve karşılaştır
                final_entity_list[i] = find_closest_word(translated, create_distance_list(text.split(" "), len(translated.split(" "))))
        else:
            final_entity_list[i] = i  # Eğer metinde bulunuyorsa, şirket adını olduğu gibi ekle
    return final_entity_list
