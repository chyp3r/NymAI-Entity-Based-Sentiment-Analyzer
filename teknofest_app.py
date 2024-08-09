import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from main import MainModel

app = FastAPI()

class Item(BaseModel):
    """
    API'nin alacağı veri modelini tanımlar.
    
    Attributes:
        text (str): İşlenecek metni temsil eden zorunlu bir alan.
    """
    # Testler için değiştirilecek satır
    text: str = Field(..., example="""Türk Başkonsolosluğu konuyla yakından ilgileniyor. Türkiye Teknoloji Takımı kötü bir hal aldı. Futbol Kulüpler Birliği yolun sonuna geldi""")

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    """
    Verilen metni işleyip modelden tahmin sonuçlarını döndüren bir API uç noktası.
    
    Args:
        item (Item): API'ye gönderilen verileri içeren model.
        
    Returns:
        dict: Modelin tahmin sonuçlarını içeren bir sözlük.
    """
    # Modeli başlat
    my_model = MainModel()
    
    # Modeli verilerle çalıştır
    my_model.execute_model(item.text)
    
    # Modelden sonuçları al
    result = my_model.take_outputs()
    
    return result

if __name__ == "__main__":
    # FastAPI uygulamasını başlat
    uvicorn.run(app, host="0.0.0.0", port=8042)
