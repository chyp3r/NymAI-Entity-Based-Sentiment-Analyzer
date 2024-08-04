import os
import logging
from typing import Callable, List

import transformers
from google.cloud.exceptions import NotFound

from . import utils
from .data_types import LabeledExample
from .models import BertABSCConfig, BertABSClassifier
from .pipelines import Pipeline
from .professors import Professor
from .aux_models import ReferenceRecognizer, PatternRecognizer

# Logger ayarları
logger = logging.getLogger('absa.load')
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DOWNLOADS_DIR = os.path.join(ROOT_DIR, 'downloads')

def load(
        name: str = 'absa/classifier-rest-0.2',
        text_splitter: Callable[[str], List[str]] = None,
        reference_recognizer: ReferenceRecognizer = None,
        pattern_recognizer: PatternRecognizer = None,
        **model_kwargs
) -> Pipeline:
    """
    Modeli ve gerekli bileşenleri yükler ve bir Pipeline oluşturur.

    Args:
    name (str): Yüklenecek modelin adı.
    text_splitter (Callable[[str], List[str]]): Metni parçalayan fonksiyon.
    reference_recognizer (ReferenceRecognizer): Referans tanıma modeli.
    pattern_recognizer (PatternRecognizer): Desen tanıma modeli.
    **model_kwargs: Modelin diğer parametreleri.

    Returns:
    Pipeline: Yüklenen model ve bileşenleri içeren Pipeline.
    """
    try:
        # Model ve tokenizer'ı yükleme
        config = BertABSCConfig.from_pretrained(name, **model_kwargs)
        model = BertABSClassifier.from_pretrained(name, config=config)
        tokenizer = transformers.BertTokenizer.from_pretrained(name)
        
        # Professor ve Pipeline oluşturma
        professor = Professor(reference_recognizer, pattern_recognizer)
        nlp = Pipeline(model, tokenizer, professor, text_splitter)
        return nlp

    except EnvironmentError as error:
        # Hata durumunda loglama ve hata fırlatma
        text = 'Model veya Tokenizer bulunamadı.'
        logger.error(text)
        raise error

def load_examples(
        dataset: str = 'semeval',
        domain: str = 'laptop',
        test: bool = False
) -> List[LabeledExample]:
    """
    Örnek veri setini yükler.

    Args:
    dataset (str): Veri seti adı.
    domain (str): Alan adı.
    test (bool): Test seti mi?

    Returns:
    List[LabeledExample]: Yüklenen örnekler.
    """
    split = 'train' if not test else 'test'
    name = f'classifier-{dataset}-{domain}-{split}.bin'
    local_path = os.path.join(DOWNLOADS_DIR, name)

    try:
        # Dosyayı buluttan indirme
        local_path = utils.file_from_bucket(name)
        examples = utils.load(local_path)
        return examples

    except NotFound as error:
        # Dosya bulunamazsa ve varsa yerel dosyayı silme
        if os.path.isfile(local_path):
            os.remove(local_path)
        text = 'Veri seti bulunamadı.'
        logger.error(text)
        raise error
