from transformers import pipeline
import re


class PrepareModel:
    def __init__(self):
        self.model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
    
    def __call__(self,text:str):
        clean_text = re.sub(r'<[^>]*>', '', text)
        return self.model(clean_text), clean_text

model = PrepareModel()