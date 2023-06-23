import cv2
import nest_asyncio
import asyncio
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
# from fastapi.responses import StreamingRespons
from fastapi import UploadFile, File
import os
from starlette.responses import JSONResponse
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
# response_model이라는 인자로 손쉽게 데이터를 정형화 가능
from pydantic import BaseModel
import detect as dt
import torch
import os
import cv2
import shlex
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import matplotlib.pyplot as plt


app = FastAPI()
nest_asyncio.apply()
# HTML 문서가 있는 폴더
templates = Jinja2Templates(directory="templates")

# CORS(app) 차단 
origins = ["*"]

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#위의 설정은 모든 origin, 모든 cookie, 모든 method, 모든 header를 allow한다.

# 내보낼 값 양식
class Item(BaseModel):
    name: str = None
    # 원하는 값 리스트로 저장
    # val: Optional[list[str]] = None
    message:  Optional[str] = None
    resres:  Optional[str] = None

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item

# 초기 화면
@app.get('/')
async def index(req : Request):
    return templates.TemplateResponse("home.html", {"request" : req})
# 캡쳐된 이미지를 실시간으로 전송

# 결과값을 받아와보자 -> 응 안돼
result = ""
def set_res(re):
    global result
    result = re
    return result

# torch model 생성, 모델을 생성해두면 속도가 훨씬 빠르다!!
house_model = torch.load('./yolomodel/hubtmp.pth')
# house_model = torch.load('./yolomodel/robotmp.pth')

# 시간 체크
import math
import time

# 쿼리 스트링으로 아이디값 받아오기(이미지 저장 문제)
#, response_model=Item, response_model_exclude_unset=Tru
@app.post('/h_photo/{cate_seq}', response_model=Item, response_model_exclude_unset=True)
# 집 1, 나무 2, 사람 3
async def upload_photo(file: UploadFile, cate_seq : int = 0):
    UPLOAD_DIR = "./images"  # 이미지를 저장할 서버 경로
    # 시간 체크
    start = time.time()
    math.factorial(100000)
    # 집, 나무, 사람 체크
    if cate_seq:
        cate_seq = cate_seq
    print(cate_seq)
    
    
    # 파일 없으면 메세지 파일 없어..
    if not file:
        items = {
            "message" : "No upload file sent"
        }
        return items
    
    # 파일 있으면
    else:
        # 이미지 받아서 저장
        content = await file.read()
        
        # filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
        # 테스트용, 중복 저장..!
        filename = "test.jpg"
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
            fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
        
        # 이미지 불러오기
        test_img=UPLOAD_DIR+"/"+filename

        """
        # 최적의 가중치 파일        
        weights="./hub0612best.pt"
        # 정확도
        conf=0.6
        img_size=640

        
        # 모델 객체 감지
        # 쉘에서 파이썬 명령 실행문
        def run_yolo_detection(weights, conf, img_size, test_img):
            # detect.py를 실행하는 명령어를 생성합니다.
            command = f"py detect.py --weights {weights} --conf {conf} --img-size {img_size} --source {test_img}"
            
            # 명령어를 실행하고 결과를 반환합니다.
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # 결과를 파일에 저장합니다.
            # with open(output_file, 'w') as f:
            #     f.write(result.stdout)
            set_res(result.stdout)
            print("checkkkkkk===================\n", result.stdout)
        
        run_yolo_detection(weights, conf, img_size, test_img)
        # please = return_res("end")
        
        plea = result.split("\n")
        
        # 원하는 정보 있는 문장
        please = plea[10]
        # 문장에서 원하는 값 추출
        aa = please.split(',')[:-2]
        
        h_robo = ['chimney', 'door', 'home', 'roof', 'sun', 'wall', 'window', 'smoke']
        
        res1=[]
        for i in aa:
            res1.append(i.split(' '))
        # resres=[]
        resres=''
        remove_set=[]
        for num,val in res1:
            # resres.append(val)
            # resres.append(num)
            # 객체값만 ','로 연결하여 문자열로 반환
            remove_set.append(val)
            resres+=val+','

        resres=resres[:-1]
        resres+=';'
        
        # 중복제거
        res_result = [i for i in h_robo if i not in remove_set]

        for i in res_result:
            resres+=i+','

        # 마지막 , 제거
        resres=resres[:-1]

        print(resres)
        """
    
        
        # 값 안받아와짐... 왜지..?
        # res = dt.return_res()
        # print(dt.return_res())
        # res = plea[-1]
        # print(res)

        
        # TEST==============================================================
        
        # model = house_model
        
        # 집 모델
        if cate_seq == 1:
            model = house_model
        # 나무, 사람 추후 업뎃
        # elif cate_seq == 2:
            # model = tree_model
        
        results = model(test_img)

        df = results.pandas().xyxy[0]
        
        # ====================== 이미지 저장 ======================================
        
        house = []

        # (bbox 그리기)
        # 결과의 위치값 추출 
        for i in range(len(df)):
            h_list = []
            x,y,w,h = list(df.iloc[i,:-3])
            label = df['name'][i]
            # 정확도
            conf = df['confidence'][i]
            x=int(x)
            y=int(y)
            w=int(w)
            h=int(h)
            # 정확도 0.5 이상만 추출
            if conf >=0.5:
                h_list.append(x)
                h_list.append(y)
                h_list.append(w)
                h_list.append(h)
                h_list.append(label)
                house.append(h_list)
        
        
        def pil_draw_rect_with_label(image, x, y, w, h, text, font_color=(255, 255, 255)):
            draw = ImageDraw.Draw(image)
            draw.rectangle((x,y,w,h), outline=(0, 255, 0), width=3)

            imageFont = ImageFont.load_default()
            text_width, text_height = imageFont.getsize(text) 
            draw.rectangle(((x, y - text_height), (x + text_width, y)), fill=(0, 0, 255)) #채워진 사각형
            draw.text((x, y - text_height), text, font=imageFont, fill=font_color)

            return image

        image = Image.open(test_img).convert('RGB')

        for i in range(len(house)):
            x,y,w,h,label = house[i]
            
            imageFont = ImageFont.load_default()
            text_width, text_height = imageFont.getsize(label)
            # draw.rectangle((x,y,w,h), outline=(0,255,0), width = 3)
            image = pil_draw_rect_with_label(image, x,y,w,h, label)

        # 이미지 사이즈 조정
        x , y = image.size
        if x > 650 or y > 650:
            # a,b= x-(x-650), y-(y-650)  # 650 사이즈 맞추기
            x,y= x-(x-(x*0.5)), y-(y-(y*0.5))
            image = image.resize((int(x),int(y)))
        
        # 파일 저장
        save_file = "./img_bbox.jpg"
        image.save(save_file,"JPEG")

        # ====================== 이미지 저장 끝 ======================================
        
        # 총 라벨
        h_robo = ['chimney', 'door', 'house', 'roof', 'sun', 'wall', 'window', 'smoke']
        # 답데이터 담을 배열
        remove_set=[]
        # 정확도 0.6 이상만 추출
        for i in range(len(df)):
            # 라벨명
            label = df['name'][i]
            # 정확도
            conf = df['confidence'][i]
            if conf >=0.6:
                remove_set.append(label)
        
        
        # 중복제거
        remove_set = set(df['name'])
        # 리스트 변환, 출력 ㅎ
        remove_set = list(remove_set)
        
        # 굴뚝 + 연기 => c_smoke로 변경
        if 'chimney' in remove_set and 'smoke' in remove_set:
            remove_set.remove('chimney')
            remove_set.remove('smoke')
            h_robo.remove('chimney')
            h_robo.remove('smoke')
            
            remove_set.append('c_smoke')
        
        # 결과값 문자열로 보낼거임!
        res = ''
        
        # ,로 결과 합치기
        for i in range(len(remove_set)):
            res += remove_set[i] + ","
        # 마지막 , 제거
        res=res[:-1]
        # 없는 객체와 구분 위해 ; 추가
        res+=';'
        
        # 중복 라벨 제거
        res_result = [i for i in h_robo if i not in remove_set]
        # 없는 라벨 객체 뒤에 ,로 연결
        for i in res_result:
            res+=i+','

        # 마지막 , 제거
        res=res[:-1]
        end = time.time()


        print("최종최종", res)
        print(f"{end - start:.5f} sec")
        items = {
            # "name" : user_id,
            "message" : res
            # "message" : "test"
        }
    return items


# 값 받는 주고받기
@app.get("/uploadfile/{index}")
# async def create_upload_file(index, file: UploadFile | None = None):
async def create_upload_file(index):
    print(index) 
    # if not file:
    #     return {"message": "No upload file sent"}
    # else:
    #     # return {"filename": file.filename}
    #     return {"filename": index}

@app.get("/test")
# 만약 값이 없을 경우 default 0
async def create_upload_file(user_id: int = 0):
        
    # 쿼리스트링으로 보낸 값 받는방법
    # user_id = request.args.get('id')
    print(user_id)
    return "dd"


# 웹 서버 설정
if __name__ == "__main__":
    # uvicorn.run(app, host="192.168.56.1", port=9000)
    uvicorn.run(app, host="211.105.164.246", port=9000)

