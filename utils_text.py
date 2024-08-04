import re
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords

def simple_stem(word):
    """
    Basit bir kök çıkarıcı fonksiyon.
    Bu fonksiyon, verilen kelimedeki kesme işareti (') karakterini kaldırır.
    
    Args:
        word (str): İşlenecek kelime.
        
    Returns:
        str: Kesme işareti kaldırılmış kelime.
    """
    splited_word = word.split("'")
    word = splited_word[0]
    return word

def clean_text(tweet): 
    """
    Tweet metnini temizler ve işler.
    URL'leri, durak kelimeleri ve özel karakterleri kaldırır.
    
    Args:
        tweet (str): İşlenecek tweet metni.
        
    Returns:
        str: Temizlenmiş tweet metni.
    """
    # İngilizce durak kelimelerini yükle
    stop_words = set(stopwords.words('english'))
    
    # URL'leri kaldır
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    
    # Durak kelimeleri ve kök çıkarıcı işlemi uygula
    tweet = ' '.join([simple_stem(word) for word in tweet.split() if word not in stop_words])
    
    # Özel karakterleri kaldır
    tweet = tweet.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?[\\]^`{|}~'))
    
    return tweet

def space_handler(text):
    """
    Metindeki fazla boşlukları ve yeni satır karakterlerini düzenler.
    
    Args:
        text (str): İşlenecek metin.
        
    Returns:
        str: Düzeltilmiş metin.
    """
    text = re.sub(" +", " ", text)  # Birden fazla boşluğu tek boşluğa çevir
    text = text.replace("\n","")   # Yeni satır karakterlerini kaldır
    return text

def translate_to_en(text):
    """
    Verilen metni otomatik olarak algılanan dilden İngilizceye çevirir.
    
    Args:
        text (str): Çevrilecek metin.
        
    Returns:
        str: İngilizce çevirisi yapılmış metin.
    """
    text = GoogleTranslator(source='auto', target='en').translate(text)
    return text

def translate_to_tr(text):
    """
    Verilen metni otomatik olarak algılanan dilden Türkçeye çevirir.
    
    Args:
        text (str): Çevrilecek metin.
        
    Returns:
        str: Türkçe çevirisi yapılmış metin.
    """
    text = GoogleTranslator(source='auto', target='tr').translate(text)
    return text

def capitalize_first_letter(sentence):
    """
    Cümledeki her kelimenin ilk harfini büyük yapar.
    
    Args:
        sentence (str): İşlenecek cümle.
        
    Returns:
        str: İlk harfleri büyük yapılmış cümle.
    """
    words = sentence.split()
    capitalized_words = [word[0].upper() + word[1:] for word in words]
    return ' '.join(capitalized_words)

def output_formater(sentimens, aspects):
    """
    Duygu ve öğe listesini formatlar ve bir sonuç sözlüğü döndürür.
    
    Args:
        sentimens (list): Duygu nesnelerinin listesi.
        aspects (dict): Öğeler ve onların isimlerini içeren sözlük.
        
    Returns:
        dict: Formatlanmış sonuç sözlüğü.
    """
    final = {}
    final["entity_list"] = list(map(lambda x: simple_stem(x.strip()), list(aspects.values())))
    
    result_list = []
    for i in sentimens:
        result_list.append({
            "entity": simple_stem(aspects[i.aspect].strip()),
            "sentiment": sentiment_formater(i.sentiment)
        })
    
    final["results"] = result_list
    return final

def sentiment_formater(sentiment):
    """
    Duygu değerini formatlar ve metin olarak döndürür.
    
    Args:
        sentiment (int): Duygu indeks değeri (0: Nötr, 1: Olumsuz, 2: Olumlu).
        
    Returns:
        str: Formatlanmış duygu metni.
    """
    sentiment_list = ["Nötr", "Olumsuz", "Olumlu"]
    try:
        return sentiment_list[sentiment]
    except Exception as e:
        print(f"Hata: {e}")
        return None