# NymAI: Varlık Tabanlı Duygu Analizcisi

## English Version
To change the English version of these documents, you can review the [README_EN.md](https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer/blob/main/README_EN.md) file.

## Proje Genel Bakışı
NymAI, Gökdeniz Kuruca tarafından "NymAI" proje adı altında geliştirilen gelişmiş bir duygu analizi aracıdır. Bu proje, Teknofest 2024 Türkçe Doğal Dil İşleme (NLP) yarışmasına katılmak üzere bir aylık bir süre içinde titizlikle hazırlanmıştır.

Bu projenin temel amacı, metinsel verilerdeki çeşitli varlıklarla ilişkili duyguları analiz etmek, gelişmiş doğruluk ve etkinlik için son teknoloji BERT tabanlı modellerden yararlanmaktır. Çözüm, adlandırılmış varlık tanıma (NER) ile duygu analizini birleştirerek karmaşık NLP görevlerini ele almak üzere tasarlanmıştır ve böylece metinlerde belirtilen belirli varlıklarla ilgili duygulara ilişkin değerli içgörüler sağlar.

### Temel Özellikler
- **Varlık Tabanlı Duygu Analizi**: Metinden çıkarılan varlıklara özel olarak odaklanan duygu analizini gerçekleştirmek için BERT tabanlı mimariyi kullanır. - **Gelişmiş Adlandırılmış Varlık Tanıma**: Kuruluşlar, konumlar ve daha fazlası gibi varlıkları tanımlamak ve kategorize etmek için sağlam NER teknikleri kullanır.
- **Çoklu Dil Desteği**: Hem İngilizce hem de Türkçe'de çeviri ve metin işlemeyi destekler. Diğer dillerle uyumludur ve tüm diller arasında geçiş yapılabilir.
- **Özelleştirilebilir ve Genişletilebilir**: Çeşitli duygu sınıflarını ve varlık türlerini barındıracak şekilde esneklikle tasarlanmıştır.

### Mimari
- **Veri Ön İşleme**: Metin verilerinin temizlenmesi ve normalleştirilmesi, özel karakterlerin işlenmesi ve gerektiğinde metnin çevrilmesi işlemlerini içerir.
- **Varlık Çıkarımı**: En son NER tekniklerini kullanarak varlıkları tanımlar ve işler.
- **Duygu Analizi**: Tanımlanan varlıklarla ilişkili duyguları analiz etmek ve sınıflandırmak için BERT tabanlı modelleri kullanır.
- **Çıktı Biçimlendirme**: Duygu sınıflandırmalarını ve ilgili varlık bilgilerini içeren yapılandırılmış çıktı sağlar.

## Kurulum
1. **Depoyu Klonla**:
    ```bash
    git clone https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer
    cd your-repository-directory

2. **Bağımlılıkları Yükle**:
    ```bash
    python -m venv env
    source env/bin/activate # Windows'ta `env\Scripts\activate` kullanın
    pip install -r requirements.txt

3. **Spacy Modellerini Yükle**:
    ```bash
    python -c "from utils import ensure_en_core_web; ensure_en_core_web('en_core_web_sm')"
    python -c "from utils import ensure_en_core_web; ensure_en_core_web('en_core_web_lg')"

## Örnek Kullanım
- Duygu analizini eylem halinde görmek için, MainModel'in işlevselliğini göstermek için örnek bir uygulama görevi gören teknofest_app.py dosyasını çalıştırın:

    ```bash
    python teknofest_app.py

- Bu betik bir FastAPI uygulamasını başlatacaktır. Analiz edilecek metni içeren bir JSON yüküyle /predict/ uç noktasına bir POST isteği göndererek duygu analizini test edebilirsiniz.

- Curl kullanarak bunu şu şekilde test edebilirsiniz::

    ```bash
    curl -X 'POST' \
    'http://127.0.0.1:8000/predict/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": Türkiye Teknoloji Takımı tarafında düzenlenen bu yarışma harika}'

- Ayrıca post isteğini http://127.0.0.1:8000/docs üzerinden de çalıştırabilirsiniz.

## Model Mimarisi
### Genel Bakış
NymAI-Entity-Based-Sentiment-Analyzer, Varlık Tabanlı Duygu Analizi (EBSA) için gelişmiş bir BERT tabanlı mimari kullanır. Metindeki belirli yönlere veya özelliklere odaklanan geleneksel Yön Tabanlı Duygu Analizi'nin (ABSA) aksine, EBSA metinde belirtilen belirli varlıklarla (şirketler veya ürünler gibi) ilişkili duyguyu vurgular.

### Bileşenler
1. **Metin Ön İşleme**
- **Metin Temizleme**:
    - **URL Kaldırma**: Analizi etkilemesi muhtemel olmayan bilgilerin önlenmesi için URL'leri ve diğer web bağlantılarını ortadan kaldırır.
    - **Özel Karakter Kaldırma**: Duygu analizine katkıda bulunmayan özel karakterleri ve noktalama işaretlerini kaldırır.
    - **Boşluk İşleme**: Boşlukları normalleştirir ve metni temizlemek için gereksiz boşlukları kaldırır. 
    - **Çeviri**: Duygu analizinde tutarlılığı sağlamak için, translate_to_en gibi çeviri işlevlerini kullanarak İngilizce olmayan metni İngilizceye çevirir.
    - **Normalleştirme**: Şirket adlarının veya jargonların farklı biçimleri gibi tutarsızlıkları ele alarak metni standartlaştırır.

2. **Adlandırılmış Varlık Tanıma (NER)**
- **Varlık Çıkarımı**:
    - **SpaCy Modelleri**: Metindeki adlandırılmış varlıkları algılamak için spaCy'nin en_core_web_sm ve en_core_web_lg modellerini kullanır. Bu varlıklar genellikle kuruluşların, ürünlerin ve diğer ilgili varlıkların adlarını içerir.
    - **Varlık Normalleştirme**: Tanımlanmış varlıkları tutarlı bir biçime dönüştürür. Bu, önekleri kaldırmayı (örneğin, '@') ve varlık adlarını standartlaştırmayı içerir.

3. **Varlık Tabanlı Duygu Analizi (EBSA)**
- **BERT Tabanlı Model**:
    - **Bağlamsal Anlama**: Bir cümledeki hem sol hem de sağ bağlamı dikkate alarak kelimelerin bağlamını yakalayan BERT'i (Transformatörlerden Çift Yönlü Kodlayıcı Gösterimleri) kullanır.
    - **Duygu Sınıflandırması**: BERT tabanlı model, her tanımlanmış varlığa özgü duyguları sınıflandırmak için ince ayarlanmıştır. Duygular genellikle olumlu, olumsuz ve nötr gibi kategorilere ayrılır.
- **Duygu Puanlaması**:
    - BERT modeli tarafından sağlanan çevreleyen metin ve bağlama göre her varlığa duygu puanları atar.

4. **Çıktı Biçimlendirmesi**
- **Sonuç Toplama**:
    - **Varlık-Duygu Eşlemesi**: Her tanımlanmış varlığı duygu puanına eşler ve metinde farklı varlıkların nasıl algılandığına dair net bir anlayış sağlar. 
- **Yapılandırılmış Çıktı**:
    - Sonuçları, ilişkili duygu sınıflandırmalarıyla birlikte varlıkların bir listesini içeren yapılandırılmış bir sözlük biçimine biçimlendirir.

### Ayrıntılı İş Akışı
1. **Giriş Metni**:
    - Analiz, şirketler veya ürünler gibi çeşitli varlıklardan bahseden ham metinle başlar.

2. **Ön İşleme**:
- **Temiz Metin**:
    - URL'leri ve özel karakterleri kaldırın.
    - Gerekirse boşlukları normalleştirin ve İngilizce olmayan metni İngilizceye çevirin.
- **Varlık Çıkarımı**:
    - Adlandırılmış varlıkları metinden algılamak ve çıkarmak için spaCy modellerini kullanın.
    - Tutarlılığı sağlamak için çıkarılan varlıkları normalleştirin.

3. **Duygu Analizi**:
    - **İşlenmiş Metni Besleyin**: Temizlenmiş ve önceden işlenmiş metni, tanımlanan varlıklarla birlikte BERT tabanlı EBSA modeline sağlayın.
    - **Duyguları Sınıflandırın**: Model, her varlıkla ilişkili duyguyu, bağlamsal anlayışına göre değerlendirir.

4. **Biçimlendirme ve Çıktı**:
    - **Sonuçları Biçimlendirin**: Her varlığı ve duygu puanını içeren yapılandırılmış bir çıktı oluşturun. 
    - **Sonuçları Döndür**: Sonuçları sözlük biçiminde, varlıkları ve bunlara karşılık gelen duygu sınıflandırmalarını listeleyerek sağlayın.

### Model Özeti
- **Metin Ön İşleme**: Giriş metninin temiz, normalleştirilmiş ve analize hazır olmasını sağlar.
- **Varlık Çıkarımı**: Gelişmiş spaCy modelleri kullanarak varlıkları tanımlar ve normalleştirir.
- **Varlık Tabanlı Duygu Analizi**: Her varlıkla ilişkili duyguyu analiz etmek ve sınıflandırmak için BERT tabanlı bir model kullanır.
- **Çıktı Biçimlendirme**: Sonuçları her varlık için duyguyu ayrıntılı olarak açıklayan net ve yapılandırılmış bir biçimde sunar.
- Bu mimari, belirli varlıkların metinde nasıl algılandığına dair doğru ve eyleme geçirilebilir içgörüler sunmak için gelişmiş metin ön işlemeyi güçlü BERT tabanlı duygu analiziyle birleştirir.

## Veri kümeleri
Modelin eğitimi ve değerlendirilmesi için kullanılan veri kümeleri SemEval14, SemEval15 ve SemEval16'dan alınmıştır. Bu veri kümeleri duygu analizi alanında yaygın olarak kullanılan ölçütlerdir ve çeşitli alanlardan açıklamalı incelemeler içerir. Veri kümeleri CSV formatında sağlanır ve projenin veri kümesi klasöründe bulunur.

1. **SemEval14**: Dizüstü bilgisayar ve restoran alanlarından gelen yorumları içerir. Her yorum, yönler ve karşılık gelen duygularla açıklanmıştır.
    - **Dizüstü Bilgisayar Yorumları**:
    Yorum sayısı: 3845
    Yön sayısı: 3045
    - **Restoran Yorumları**:
    Yorum sayısı: 3300
    Yön sayısı: 3813

2. **SemEval15**: Veri setini daha fazla yorumla genişletir ve oteller ve cihazlar gibi ek alanlar içerir.
    - **Otel Yorumları**:
    Yorum sayısı: 2000
    Yön sayısı: 1692
    - **Cihaz Yorumları**:
    Yorum sayısı: 1315
    Yön sayısı: 2448

3. **SemEval16**: Daha ayrıntılı yön kategorilerine ve ek duygu sınıflarına odaklanarak veri setini daha da genişletir. - **Restoran İncelemeleri**:
    İnceleme sayısı: 2000
    Yön sayısı: 1743
    - **Dizüstü Bilgisayar İncelemeleri**:
    İnceleme sayısı: 3000
    Yön sayısı: 2949

    Her veri kümesi tipik olarak şunları içeren sütunlar içerir:

    text: İncelemenin metni.
    aspect: İncelenen belirli varlık veya özellik.
    sentiment: Yöne yönelik ifade edilen duygu (örn. olumlu, olumsuz, nötr).

Bu veri kümeleri, BERT tabanlı varlık tabanlı duygu analizi modelinin eğitimi için çok önemlidir ve çeşitli incelemelerden öğrenmesini ve farklı varlıklar ve alanlar genelinde duyguları doğru bir şekilde tahmin etmesini sağlar.

Veri kümesi deneyleri sırasında, bu veri kümeleri Türkçeye de çevrildi ve eğitim için kullanıldı. Ancak, Türkçe veri kümelerinden elde edilen sonuçlar istenen performans seviyelerini karşılamadı. Sonuç olarak, veri kümelerinin orijinal dillerinde tutulmasına ve bunun yerine işlenecek cümlelerin çevrilmesine karar verildi. Bu yaklaşım, duygu analizinde daha iyi doğruluk ve tutarlılık sağlar.

## Farklı Varlık Türleri için Adlandırılmış Varlık Tanıma'yı (NER) Özelleştirme
ner.py dosyasında, NER sisteminin varsayılan uygulaması metinden kuruluş adlarını (ORG olarak etiketlenir) tanımlamaya odaklanır. Farklı varlık türlerini (örneğin kişiler için PERSON, konumlar için LOC vb.) çıkarmanız gerekiyorsa, bu değişiklikleri karşılamak için NER sistemini değiştirmeniz gerekecektir.

Farklı varlık türlerine odaklanmak için varlık çıkarmayı nasıl özelleştirebileceğiniz aşağıda açıklanmıştır:

1. **Varlık Çıkarma Kodunu Bulun**: ner.py'de varlıkların SpaCy modelleri kullanılarak çıkarıldığı satırları bulun:

    ```bash
    companies_sm = [ent.text.replace("@", "") for ent in doc_sm.ents if ent.label_ == "ORG"]
    companies_lg = [ent.text.replace("@", "") for ent in doc_lg.ents if ent.label_ == "ORG"]

2. **Varlık Etiketini Değiştirin**: ORG'yi ilgilendiğiniz varlık türü için uygun etiketle değiştirin. SpaCy, farklı varlık türleri için çeşitli etiketler sağlar. Kullanabileceğiniz bazı yaygın etiketler şunlardır:

    PERSON: Kurgusal olanlar da dahil olmak üzere insanlar.
    NORP: Milliyetler veya dini veya siyasi gruplar. FAC: Binalar, havaalanları, otoyollar, köprüler, vb.
    ORG: Şirketler, ajanslar, kurumlar, vb.
    GPE: Ülkeler, şehirler, eyaletler.
    LOC: GPE dışı konumlar, dağ sıraları, su kütleleri.
    PRODUCT: Nesneler, araçlar, yiyecekler, vb. (hizmetler değil).
    EVENT: Adlandırılmış kasırgalar, savaşlar, savaşlar, spor etkinlikleri, vb.
    WORK_OF_ART: Kitap, şarkı, vb. başlıkları
    LAW: Yasa haline getirilmiş adlandırılmış belgeler.
    LANGUAGE: Herhangi bir adlandırılmış dil.

3. **Örnek Değişiklik**: ORG yerine PERSON varlıklarını çıkarmak için satırları aşağıdaki gibi değiştirirsiniz:

4. **Değişiklikleri Kaydet**: Gerekli değişiklikleri yaptıktan sonra [ner.py](https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer/blob/main/ner.py) dosyasını kaydedin.

5. **Diğer Bileşenleri Güncelleyin**: Bu varlıklara bağlı olan kodunuzun diğer bölümlerinin yeni varlık türünü işlemek için uygun şekilde güncellendiğinden emin olun.

    Bu adımları izleyerek, varlık çıkarma sürecini uygulamanızla ilgili belirli varlık türlerine odaklanacak şekilde özelleştirebilirsiniz. Bu esneklik, NER sistemini kuruluşlara yönelik varsayılan odaklanmanın ötesinde çok çeşitli kullanım durumlarına uyarlamanıza olanak tanır.

## Lisans
Bu proje Apache Lisansı 2.0 altında lisanslanmıştır. Ayrıntılar için [LICENSE](https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer?tab=Apache-2.0-1-ov-file) dosyasına bakın.

## Atıf
Bu kodu veya herhangi bir bölümünü araştırmanızda kullanırsanız, lütfen çalışmamızı atıflamayı düşünün:

    @misc{nymAI2024,
    author = {Gökdeniz Kuruca},
    title = {NymAI-Entity-Based-Sentiment-Analyzer},
    year = {2024},
    howpublished = {\url{https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer}},
    note = {Teknofest 2024 Türk Doğal Dil İşleme yarışması için Alzcur ekibi tarafından geliştirilmiştir.}
    }


## Teşekkürler

Bu projenin kodunun bazı bölümleri [Aspect-Based Sentiment Analysis](https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis) deposundan uyarlanmıştır. Bu projenin geliştirilmesinde etkili olan katkılarından dolayı bu deponun yazarlarına şükranlarımızı sunarız.

Kullanım ve dağıtım terimleri hakkında ayrıntılar için lütfen [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)'a bakın.

    
