<h1 align="center"><a href='https://www.notion.so/1f41f1722e824bca82b61eb4240f0356?pvs=4'>RecSys-03 ㄱ해줘</a></h1>
<br></br>

## 🏆 대회 개요 🏆

  과거에는 접근조차 어려웠던 분야에서도 인공지능과 머신러닝의 발전으로 모델이 훌륭한 성과를 내고 있다. 만약 대표적인 암호화폐인 비트코인의 가격 등락 예측이 정확하게 이뤄진다면, 투자자들의 투자 전략 수립에 큰 도움이 될 수 있고, 또한 모델 개발을 통해 얻어진 인사이트는 다른 금융 상품의 예측 모델 개발에도 기여할 수 있다.

- Objective
    - **비트코인의 다음 시점(한 시간 뒤)에서의 가격 등락 예측**

<br></br>
## 👨‍👩‍👧‍👦 팀 소개 👨‍👩‍👧‍👦
    
|강성택|김다빈|김윤경|김희수|노근서|박영균|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/TaroSin'><img src='https://github.com/user-attachments/assets/75682bd3-bcff-433e-8fe5-6515a72361d6' width='200px'/></a>|<a href='https://github.com/BinnieKim'><img src='https://github.com/user-attachments/assets/ff639e97-91c9-47e1-a0c8-a5fc09c025a6' width='200px'/></a>|<a href='https://github.com/luck-kyv'><img src='https://github.com/user-attachments/assets/015ec963-d1b4-4365-91c2-d513e94c2b8a' width='200px'/></a>|<a href='https://github.com/0k8h2s5'><img src='https://github.com/user-attachments/assets/526dc87c-0122-4829-8e94-bce6f15fc068' width='200px'/></a>|<a href='https://github.com/geunsseo'><img src='https://github.com/user-attachments/assets/0a1a27c1-4c91-4fdf-b350-1540c835ee72' width='200px'/></a>|<a href='https://github.com/0-virus'><img src='https://github.com/user-attachments/assets/98470105-260e-443d-8592-c139d7918b5e' width='200px'/></a>|

<br></br>

## 🌳 File Tree 🌳

```
{level1-classificationinmachinelearning-recsys-03}
│
├── code
│   ├── Final_Ensemble_V2.ipynb
│   ├── Final_Ensemble_test.ipynb
│   ├── data_preprocessing.py
│   └── requirements.txt
│
├── EDA # 각자 시도한 EDA 파일
│   ├── correlation_EDA.ipynb
│   ├── davin_EDA.ipynb
│   ├── geunsseo_EDA.ipynb
│   ├── hs_EDA.ipynb
│   ├── tarosin_EDA.ipynb
│   ├── yoon_EDA.ipynb 
│   └── zerovirus_EDA.ipynb
│
└── trial # 실험 파일
    ├── Baseline_code_EDA.ipynb         # EDA 베이스라인
    ├── Ensemble_all_Final.ipynb        # 파이프라인으로 피처 제공, optuna O
    ├── Ensemble_all_RandomForest.ipynb # 피처 통일, RF 추가한 앙상블
    ├── Ensemble_Stacking.ipynb         # 스태킹 앙상블
    ├── Ensemble_all_Optuna.ipynb       # 피처 통일(LGB), optuna O
    ├── Ensemble_all_Search.ipynb       # 피처 통일, optuna X 
    ├── Ensemble_each.ipynb             # 각 모델별로 피처 선택
    ├── LightGBM.ipynb                  # LGB 베이스라인
    ├── LightGBM_Optuna.ipynb           # LGB 실험
    ├── LinearRegression.ipynb          # LR 실험
    ├── RandomForest.ipynb              # RF 실험
    ├── XGBoost.ipynb                   # XGB 베이스라인
    ├── XGBoost_Optuna.ipynb            # XGB 실험
    ├── data_preprocessing.ipynb
    └── requirements.txt
```

### data_preprocessing.py: 데이터 전처리 함수가 정의되어 있는 py 파일

<br></br>

## ▶️ 실행 방법 ▶️

```python
pip install -r requirements.txt
```

<br></br>

## GitHub Convention

- `main branch`는 배포이력을 관리하기 위해 사용
  
- `BTC branch`는 기능 개발을 위한 branch들을 병합(merge)하기 위해 사용
- 모든 기능이 추가되고 버그가 수정되어 배포 가능한 안정적인 상태라면 `BTC branch`에 병합(merge)
- 작업을 할 때에는 개인의 branch를 통해 작업
- EDA
  - branch명 형식은 `EDA-자기이름` 으로 작성 ex) EDA-TaroSin
  - 파일명 형식은 `name_EDA` 으로 작성 ex) TaroSin_EDA
- 데이터 전처리팀
  - branch명의 형식 `data/기능이름` 으로 작성 ex) data/rolling, data/sort $\cdots$
  - `data_preprocessing.py` 파일 내에 함수 추가
- 모델팀
  - branch명의 형식 `model/기능이름`으로 작성 ex) model/LSTM, model/LightGBM $\cdots$
  - 파일명 형식 `모델이름_버전` 으로 작성 ex) XGBoost_optuna.ipynb
- `master(main) Branch`에 Pull request를 하는 것이 아닌, `model Branch` 또는 `data Branch`에 Pull request 요청
- commit을 할 경우 어떤 부분을 수정하였는지 작성
    ```bash
    git commit -m “added 기능이름 to data”
    git commit -m “fixed 기능이름 to data”
    git commit -m “deleted 기능이름 to data”
    git commit -m “completed 기능이름 to data”
    ```
- pull request merge 담당자 : data - 다빈 / model - 영균  (다빈, 영균은 성택이 담당) </br>*나머지는 BTC branch 건드리지 말 것!*
- Pull request는 Template 에 맞추어 작성 (커스텀 Labels 사용)

<br></br>

## Code Convention

- 문자열을 처리할 때는 큰 따옴표를 사용하도록 합니다.
- 클래스명은 `카멜케이스(CamelCase)` 로 작성합니다. </br>
  함수명, 변수명은 `스네이크케이스(snake_case)`로 작성합니다.
- 객체의 이름은 해당 객체의 기능을 잘 설명하는 것으로 정합니다.  
    ```python
    # bad
    a = ~~~
    # good
    lgbm_pred_y = ~~~
    ```
- 가독성을 위해 한 줄에 하나의 문장만 작성합니다.
- 들여쓰기는 4 Space 대신 Tab을 사용합시다.
- 주석은 설명하려는 구문에 맞춰 들여쓰기, 코드 위에 작성 합니다.
    ```python
    # good
    def some_function():
      ...
    
      # statement에 관한 주석
      statements
    ```
    
- (데이터 전처리팀) 전처리별 구분 주석은 ###으로 한 줄 위에 작성 합니다.
    
    ```python
    # good
    ### normalization
    
    def standardize_feature(
    ```
    
- 키워드 인수를 나타낼 때나 주석이 없는 함수 매개변수의 기본값을 나타낼 때 기호 주위에 공백을 사용하지 마세요.
    
    ```python
    # bad
    def complex(real, imag = 0.0):
        return magic(r = real, i = imag)
    # good
    def complex(real, imag=0.0):
        return magic(r=real, i=imag)
    ```
    
- 연산자 사이에는 공백을 추가하여 가독성을 높입니다.
    
    ```python
    a+b+c+d # bad
    a + b + c + d # good
    ```
    
- 콤마(,) 다음에 값이 올 경우 공백을 추가하여 가독성을 높입니다.
    
    ```python
    arr = [1,2,3,4] # bad
    arr = [1, 2, 3, 4] # good
    ```
