# ragMenu.py
# pip install pillow
# 이미지 처리 모듈
from langchain_openai import ChatOpenAI
import base64
from PIL import Image
import io

from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini")
#GPT Vision 지원 모델

#이미지 base64 인코딩하는 함수
def encode_image(path):
    with open(path,'rb') as f:
        obj=base64.b64encode(f.read()).decode('utf-8')
        return obj

def analyze_image(image_path):
    #base64 인코딩 처리
    img_b64=encode_image(image_path) 

    #llm에게 분석하도록 하기 위해 프롬프트 작성
    messages=[
        {
            "role":"user",
            "content":[
                {"type":"text", "text":"이 이미지를 분석해줘"},
                {
                "type": "image_url",
                "image_url":{
                        "url":f"data:image/png;base64,{img_b64}"
                    } 
                }
            ]
        }
    ]
    #llm호출
    response = llm.invoke(messages)
    print("===이미지 분석 결과=============")
    print(response.content)
    print("==============================")

if __name__ =='__main__':
    filepath='./data/wine.png'
    analyze_image(filepath)