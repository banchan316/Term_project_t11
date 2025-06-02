# US Accidents 데이터 분석 – Team 11 Term Project

## 프로젝트 개요

- **주제**: 미국 교통사고 데이터셋을 활용하여 사고 심각도 예측 및 사고 처리 시간(Duration) 예측
- **목표**
  - 사고 심각도(Severity)에 영향을 미치는 요인을 분석하고 예측 모델 개발
  - 사고 처리 시간(Duration)을 예측하고 그에 영향을 미치는 요인을 분석

## 데이터셋
- 출처: [Kaggle US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)
- 기간: 2016년 2월 ~ 2023년 3월
- 크기: 약 7.7M rows, 47 columns
- 주요 Feature
  - Severity (1~4단계)
  - Weather, Visibility, Temperature, Wind_Speed, Road Info
  - Start_Time, End_Time, Start_Lat, Start_Lng, etc.

---

## 폴더 구조

```
.
├── main.py                # 전체 실행 스크립트
├── preprocessing.py       # 결측치 처리, 인코딩 등 전처리 함수들
├── feature_scaling.py     # 스케일링 함수
├── model.py               # 모델 정의 및 평가 (roc 등 포함)
├── inspection.py          # 데이터 시각화 및 탐색 함수
├── open_source.py         # GitHub 업로드용 스크립트
└── US_Accidents_March23_sampled_500k.csv  # 샘플 데이터
```

---

## 예측 대상 (Target 변수)

- `Severity` (분류): 1,2 → 0 / 3,4 → 1 로 이진 분류로 변환
- `Duration_Minutes` (회귀): 사고 시작/종료 시간 차이로 계산

---

## 1. Data Inspection
- 범주형 변수: `Weather_Condition`, `State`, `Traffic_Signal` 등 상위 10개 값 분포 시각화
- 수치형 변수: 기본 통계 확인 및 이상치 탐지
- `Severity` 분포 확인 (2가 전체의 약 80%)
- 결측치 비율 ≥ 40% 변수 제외
- 중복 데이터 없음 확인
- Boolean 변수(`Junction`, `Stop` 등) True/False 비율 확인
- 지도 시각화 (Plotly 사용)

---

## 2. Preprocessing

### 2-1 Feature Engineering
- `Duration_Minutes` 생성: Start_Time ~ End_Time 간 시간차 (분 단위)
- 시간 파생 변수: `Start_Hour`, `Start_Month`, `Start_DayOfWeek`
- 타겟 변수 구성
  - Classification: `Severity_binary` (1,2 → 0 / 3,4 → 1)
  - Regression: `Duration_Minutes`
- 불필요한 변수 제거: `ID`, `Street`, `End_Lat`, `Country` 등

### 2-2 Handle Missing Value

#### 수치형
- 일반 컬럼: 중앙값, 평균값 또는 0으로 대체
- **Precipitation(in)**: 간소화된 날씨 조건(`Weather_Condition`) 기준 평균값으로 대체

#### 범주형
- 최빈값 또는 `unknown`으로 대체

### 2-3 One-Hot Encoding
- `Wind_Direction`, `Weather_Condition` 등 범주형 변수 OHE 적용
- 일부 범주는 `North`, `South` 등으로 간소화 후 인코딩

### 2-4 Feature Scaling
- 스케일러 비교: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- 스케일링 전후 평균, 표준편차, 최소/최대값 비교 및 시각화

### 2-5 Train/Test Split
- **StratifiedKFold (k=5)**를 사용한 K-fold 교차검증
- 클래스 불균형을 고려하여 stratified 방식 사용

---

## 3. Modeling & Evaluation

### Classification - 사고 심각도 예측
- RandomForestClassifier
- GradientBoostingClassifier
- XGBoostClassifier (scale_pos_weight 설정, threshold 조정 포함)

### Regression - 사고 처리 시간(Duration) 예측
- log1p(Duration_Minutes) 변환 후 모델 학습
- RandomForest, GradientBoosting, XGBoost 사용


## 성능 평가 지표

|     문제 유형    |                  사용 지표                     |
|----------------|---------------------------------------------|
| 분류 (Severity) | Accuracy, Weighted F1 Score (클래스 불균형 고려) |
| 회귀 (Duration) | RMSE, R² Score                              |

※ 분류 문제는 **Stratified K-Fold Cross Validation**을 적용


## 모델 성능 비교 및 Top 5 조합

### 분류 (Severity)

| 순위  |     스케일러      |       모델        |    지표    |  점수   |
|------|----------------|------------------|-----------|--------|
| 1    | StandardScaler | XGBoost          | F1 Score  | 0.7991 |
| 2    | MinMaxScaler   | XGBoost          | F1 Score  | 0.7948 |
| 3    | RobustScaler   | XGBoost          | F1 Score  | 0.7917 |
| 4    | MinMaxScaler   | GradientBoosting | F1 Score  | 0.7864 |
| 5    | StandardScaler | GradientBoosting | F1 Score  | 0.7832 |

**최종 선정 모델**: `XGBoost + StandardScaler` (F1 Score = 0.7991)


### 회귀 (Duration_Minutes)

| 순위  |      스케일러     |           모델             |  지표  |  점수   |
|------|----------------|---------------------------|-------|--------|
| 1    | RobustScaler   | XGBoostRegressor          |   R²  | 0.8536 |
| 2    | StandardScaler | RandomForestRegressor     |   R²  | 0.8478 |
| 3    | RobustScaler   | RandomForestRegressor     |   R²  | 0.8452 |
| 4    | MinMaxScaler   | XGBoostRegressor          |   R²  | 0.8410 |
| 5    | StandardScaler | GradientBoostingRegressor |   R²  | 0.8396 |

**최종 선정 모델**: `XGBoostRegressor + RobustScaler` (R² = 0.8536)

---

## 분석 해석 및 학습 경험

- 결측치 처리 방식에 따라 모델 성능이 민감하게 반응함을 확인
- One-Hot Encoding, 스케일링 방식에 따른 모델별 성능 변화가 존재
- `XGBoost`는 대부분 상황에서 강력한 성능을 보여주며 best model로 선정
- **Stratified K-Fold**를 통해 데이터 불균형의 영향을 최소화하며 안정적인 성능 평가 가능
- 협업을 통해 모듈화, 기능 분할, 리팩토링의 중요성을 학습함

---

## 오픈소스 기여

- 전처리 함수 모듈화: `preprocessing.py`, `feature_scaling.py`
- 결측치 처리 함수: `handle_missing_value()`
- 성능 비교 자동화: `main.py`, `open_source.py` 통해 다양한 조합 평가
- 주석, 타입 힌트, 실행 예시 포함 문서화
- [GitHub Repository](https://github.com/banchan316/Term_project_t11)

---

## 팀 구성 (Team 11)

| 이름  |    학번    |
|------|-----------|
| 김병규 | 202135730 |
| 이찬  | 202135815 |
| 이슬기 | 202235085 |
| 김지해 | 202334445 |
| 박지훈 | 202337621 |

---
