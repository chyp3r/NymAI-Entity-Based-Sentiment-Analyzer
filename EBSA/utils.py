import os
import pickle
import logging
from typing import Any, Iterable, List
from google.cloud import storage

logger = logging.getLogger('absa.utils')

def load(file_path: str) -> Any:
    """
    Verilen dosya yolundan pickle ile veriyi yükler.
    
    Args:
        file_path (str): Yüklemek için dosya yolu.
    
    Returns:
        Any: Yüklenen veri.
    """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)

def save(data: Any, file_path: str):
    """
    Veriyi pickle ile belirtilen dosya yoluna kaydeder.
    
    Args:
        data (Any): Kaydedilecek veri.
        file_path (str): Kaydedilecek dosya yolu.
    """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)

def batches(examples: Iterable[Any], batch_size: int, reminder: bool = True) -> Iterable[List[Any]]:
    """
    Verilen örnekleri belirtilen batch boyutuna göre parçalara böler.
    
    Args:
        examples (Iterable[Any]): Bölünecek örnekler.
        batch_size (int): Batch boyutu.
        reminder (bool): Son batch'i döndürme seçeneği.
    
    Returns:
        Iterable[List[Any]]: Batch'ler halinde örnekler.
    """
    batch = []
    for example in examples:
        batch.append(example)
        if len(batch) < batch_size:
            continue
        yield batch
        batch = []
    if batch and reminder:
        yield batch

def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """
    Belirtilen bucket'tan bir dosyayı indirir.
    
    Args:
        bucket_name (str): Bulut bucket'ının adı.
        remote_path (str): Bucket içindeki dosyanın yolu.
        local_path (str): Dosyanın indirileceği yerel yol.
    """
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = storage.Blob(remote_path, bucket)
    blob.download_to_filename(local_path, client=client)

def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """
    Eğer dosya yerel olarak mevcut değilse, bucket'tan indirir.
    
    Args:
        bucket_name (str): Bulut bucket'ının adı.
        remote_path (str): Bucket içindeki dosyanın yolu.
        local_path (str): Dosyanın indirileceği yerel yol.
    """
    if os.path.isfile(local_path):
        return
    directory = os.path.dirname(local_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('Downloading file from the bucket...')
    download_from_bucket(bucket_name, remote_path, local_path)

def file_from_bucket(name: str) -> str:
    """
    Belirtilen dosya adını kullanarak bucket'tan dosya indirir ve yerel yolu döndürür.
    
    Args:
        name (str): Dosya adı.
    
    Returns:
        str: Yerel dosya yolu.
    """
    bucket = 'aspect-based-sentiment-analysis'
    remote_path = name
    local_path = f'{os.path.dirname(__file__)}/downloads/{name}'
    maybe_download_from_bucket(bucket, remote_path, local_path)
    return local_path

def cache_fixture(fixture):
    """
    Fixture'ların cache'lenmesini sağlar.
    
    Args:
        fixture: Cache'lenmesi gereken fixture fonksiyonu.
    
    Returns:
        Callable: Cache'lenmiş fixture döndüren wrapper fonksiyonu.
    """
    def wrapper(request, *args):
        name = request.fixturename
        val = request.config.cache.get(name, None)
        if not val:
            val = fixture(request, *args)
            request.config.cache.set(name, val)
        return val

    return wrapper
