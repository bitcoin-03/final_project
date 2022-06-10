![Logo.jpg](https://user-images.githubusercontent.com/37561451/173091343-45a49d7f-4620-43a4-b504-ead506e0934d.png)

## 1️⃣ Introduction

### 1) Background

​		기존의 닮은 배우(or 연예인) 찾기 서비스가 실제 사용자와 크게 닮지 않다는 점에서 착안

### 2) Project Objective

- 사용자가 배우와 닮았다고 느낄 수 있도록 개선한 서비스를 웹으로 제공
- Efficientnet과 BeautyGAN을 사용하여 최소한의 서비스를 구현

![Untitled](https://user-images.githubusercontent.com/37561451/173113075-d798d36e-949e-49e7-9ce8-9b0c29d0ccb8.png)

------

## 2️⃣ Demo 영상

<img src="https://user-images.githubusercontent.com/37561451/173114020-0ebc9e4c-41b4-4b5c-b2bb-3b0c65efba2f.gif" width="55%" />

------

## 3️⃣ Service Architecture

### 1) Directory 구조

```shell
   final-project-level3-cv-03
   ├── 📁 Crawiling_Part⋮
   │    └──  ⋮
   ├── 📁 Model_Part
   │    └──  ⋮
   ├── 📁 Tools
   │    └──  ⋮
   ├── 📁 Web_Part
   │    ├── 📁 back_fastapi
   │    │   └── 📁 app
   │    │       ├── 💾 __main__.py
   │    │       ├── 💾 main.py
   │    │       ├── 📁 routers
   │    │       │   ├── 💾 face_classifier.py
   │    │       │   └── 💾 face_makeup.py
   │    │       ├── 💾 storage.py
   │    │       └── 💾 utils.py
   │    ├── 📁 front_streamlit
   │    │   ├── 💾 app.py
   │    │   ├── 💾 utils.py
   │    │   └──  ⋮ 
   │    ├── 📁 kakaotalk_share
   │    │    ├── 💾 __init__.py
   │    │    └── 💾 index.html
   │    ├── 📁 log
   │    │    └── ⋮ 
   │    ├── 📁 models
   │    │   ├── 📁 beautygan
   │    │   │   ├── 💾 beautygan_model.py
   │    │   │   └── 📁 weights
   │    │   └── 📁 efficientnet
   │    │       ├── 💾 efficientnet_model.py
   │    │       └── 📁 weights
   │    ├── 💾 actor.json
   │    ├── 💾 config.yaml
   │    ├── 💾 logger.py
   │    └── 💾 Makefile 
   └── 💾 requirements.txt

```

------

## 4️⃣ DataSet

### 1) 데이터셋 구성 파이프라인

![dataflow](https://user-images.githubusercontent.com/37561451/173118470-c75e7061-62f1-49b4-96a5-4af453f5facd.png)

### 2) 데이터셋 수집

- **총** **51221**장 → 배우 별 **8:2**의 비율로 **train set**과 **valid set**구성
- **mtcnn**을 이용해 얼굴이 하나만 탐지된 사진 중 일정한 화질 이상의 이미지를 수집 
- **배우 이름, 시사회, 화보**등 키워드를 검색에 이용, **배우 당 500개**의 이미지를 크롤링

------

## 5️⃣ Modeling

### 1) Flow Chart

![flowchart](https://user-images.githubusercontent.com/37561451/173102952-68b35df9-1119-45ef-bbe4-42131544915f.png)

### 2) Preprocessing

- 선글라스 착용, 2개 이상의 얼굴, 옆모습 제외
- JPEG 형식 통일
- Insight face를 사용
  - 다수의 얼굴이 detect 되는 경우 제외
  - 얼굴 부분이 너무 작거나 없는 경우 제외
- algin, crop

------

## 6️⃣ Product Serving

### 1) FrontEnd (Streamlit)

- 

### 2) BackEnd (FastAPI)

- 

### 3) Github Action

- 

### 4) Getting Started!

1. Python requirements
   `Python`: 3.7.13

2. Installation

   1. 가상 환경을 설정합니다

   2. 프로젝트의 의존성을 설치합니다

      - ```
        requirements.txt
        ```

        를 사용하여 라이브러리를 설치합니다.

        ```
        > pip install -r requirements.txt 
        ```

   3. 아래 url에 들어가서 beautygan의 가중치는 Web_Part/models/beautygan/weights 폴더 안으로, efficientnet의 가중치는 Web_Part/models/efficientnet/weights 다운받습니다.

      - beautygan 가중치 : <https://drive.google.com/drive/folders/1pgVqnF2-rnOxcUQ3SO4JwHUFTdiSe5t9>
      - efficientnet 가중치 : <https://drive.google.com/drive/folders/113pJ2YZa_AuOGWpan7qotU3s374KzpbC?usp=sharing>

   4. ```
      > cd Web_Part
      ```

   5. Frontend(Streamlit)와 Server(FastAPI)를 같이 실행합니다

      ```
      > make -j 2 run_app
      ```



------

## 7️⃣  Appendix

### 타임라인

![timeline](https://user-images.githubusercontent.com/37561451/173107042-984b7194-a7c6-43c1-a642-a70067f76e6b.png)

### 협업 Tools

- **notion**

  - notion을 활용해 **1차 기능 구현 계획**을 세우고 각 part별로 업무를 작성
  - **project kanban board**를 통해 **업무, part, 진행률, 담당자**를 명시하여 서로의 작업 상황을 공유
  - [bittcoin notion link](https://sand-bobolink-9c4.notion.site/Final-Project-0e0a8f40e20143c89e06439e6af43b9a)

  

- **github**

  - 전체적으로 **github flow**를 사용하여 repo 관리
  - **release branch**를 통해 2.0.1까지 총 4가지 버전 배포

------

## 8️⃣ 팀원 소개
