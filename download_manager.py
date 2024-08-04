import spacy
import subprocess
import sys

def ensure_en_core_web(model_name):
    # Modelin yüklenip yüklenmediğini kontrol et
    try:
        spacy.load(model_name)
        print(f"{model_name} model çoktan yüklenmiş")
    except OSError:
        print(f"{model_name} model bulunamadı. Yükleniyor...")
        
        # Modeli yüklemek için subprocess kullanarak komut çalıştır
        try:
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model_name])
            print(f"{model_name} model başarılı bir şekilde yükelndi.")
        except subprocess.CalledProcessError as e:
            print(f"{model_name} modeli için yükleme başarısız: {e}")


