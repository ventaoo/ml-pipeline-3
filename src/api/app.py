import configparser
import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import Word2Vec

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from db import store_prediction_results
from init_env import decrypt_with_ansible_lib

encrypted_file = os.getenv("DECRYPT_FILE_PATH")
encrypted_password = os.getenv("DECRYPT_PASSWORD")
output_file = os.getenv("OUTPUT_FILE")
decrypt_with_ansible_lib(encrypted_file, encrypted_password, output_file)

# 创建 FastAPI 应用
app = FastAPI()

# 定义数据输入格式
class InputData(BaseModel):
    text: str

# API 类
class api_():
    def __init__(self):
        # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        # 加载模型
        self.model = joblib.load(self.config['LOG_REG']['path'])
        self.word2vec_model = Word2Vec.load(self.config['WORD2VEC']['model_path'])

    def text_to_vector(self, model, text):
        """ 将文本转换为词向量的平均值 """
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    def predict(self, input_text):
        """ 执行预测 """
        input_data = np.array([self.text_to_vector(self.word2vec_model, input_text)])
        prediction = self.model.predict(input_data)
        return int(prediction)  # 确保返回整数

# 预测 API 实例
api_interface = api_()

# 预测接口
@app.get("/predict/{data}")
def predict(data: str):
    prediction = api_interface.predict(data)
    
    # 存入数据库时存储 (输入文本, 预测结果)
    store_prediction_results([(data, prediction)], "predictions")  
    
    return {"input": data, "prediction": prediction}
