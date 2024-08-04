def levenshtein_distance(s1, s2):
    """
    Levenshtein mesafesi hesaplar, yani iki kelime arasındaki edit mesafesi.
    Edit mesafesi, bir kelimeyi diğerine dönüştürmek için gereken ekleme, silme ve değiştirme işlemlerinin sayısını ifade eder.
    
    Args:
        s1 (str): İlk kelime.
        s2 (str): İkinci kelime.
        
    Returns:
        int: İki kelime arasındaki Levenshtein mesafesi.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)  # Daha kısa kelimeyi s1 yap
    if len(s2) == 0:
        return len(s1)  # Eğer s2 boşsa, s1'in uzunluğunu döndür
    
    # Önceki satır
    previous_row = range(len(s2) + 1)
    
    # Her karakter için satır hesaplaması
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_closest_word(word, sentence_words):
    """
    Verilen bir kelimenin, cümledeki kelimeler arasında en yakın eşleşeni bulur.
    
    Args:
        word (str): Eşleşmesi aranacak kelime.
        sentence_words (list): Cümledeki kelimelerin listesi.
        
    Returns:
        str: En yakın eşleşen kelime.
    """
    closest_word = None
    min_distance = float('inf')
    
    for sentence_word in sentence_words:
        distance = levenshtein_distance(word, sentence_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = sentence_word
    
    return closest_word

def create_distance_list(sentence_words, count):
    """
    Cümledeki kelimelerden belirli uzunlukta alt diziler oluşturur.
    
    Args:
        sentence_words (list): Cümledeki kelimelerin listesi.
        count (int): Oluşturulacak alt dizilerin kelime sayısı.
        
    Returns:
        list: Belirtilen uzunlukta kelime dizileri içeren liste.
    """
    return_list = []
    for i in range(0, len(sentence_words) - count + 1):
        temp = ""
        for b in range(i, i + count):
            temp += sentence_words[b] + " "
        return_list.append(temp.strip())  # Boşluklardan arındırılmış dizi
    
    return return_list
