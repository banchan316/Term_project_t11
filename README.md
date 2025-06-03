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
- 샘플 데이터: 500k rows (분석용)
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
├── model.py               # 모델 정의 및 평가 (ROC, PR 곡선 포함)
├── inspection.py          # 데이터 시각화 및 탐색 함수
├── open_source.py         # GitHub 업로드용 스크립트 (개선된 교차검증)
└── US_Accidents_March23_sampled_500k.csv  # 샘플 데이터
```

---

## 예측 대상 (Target 변수)

- `Severity` (분류): 1,2 → 0 / 3,4 → 1 로 이진 분류로 변환
- `Duration_Minutes` (회귀): 사고 시작/종료 시간 차이로 계산

---

## 1. Data Inspection
- 범주형 변수: `Weather_Condition`, `State`, `Wind_Direction` 등 상위 10개 값 분포 시각화
- 수치형 변수: 기본 통계 확인 및 이상치 탐지
- `Severity` 분포 확인 (2가 전체의 약 80% - 클래스 불균형 존재)
- 결측치 비율 ≥ 40% 변수 제외 (`End_Lat`, `End_Lng`, `Wind_Chill(F)` 등)
- 중복 데이터 없음 확인
- Boolean 변수(`Junction`, `Stop`, `Traffic_Signal` 등) True/False 비율 확인
- 지도 시각화 (Plotly 사용하여 미국 전역 사고 분포 확인)

---

## 2. Preprocessing

### 2-1 Feature Engineering
- `Duration_Minutes` 생성: Start_Time ~ End_Time 간 시간차 (분 단위)
- 음수 Duration 및 7일 초과 이상치 제거
- 시간 파생 변수: `Start_Hour`, `Start_Month`, `Start_DayOfWeek`
- 타겟 변수 구성
  - Classification: `Severity_binary` (1,2 → 0 / 3,4 → 1)
  - Regression: `Duration_Minutes`
- 불필요한 변수 제거: `ID`, `Street`, `End_Lat`, `Country`, `Description` 등

### 2-2 Handle Missing Value

#### 수치형
- 일반 컬럼: 중앙값, 평균값 또는 0으로 대체
- **Precipitation(in)**: 간소화된 날씨 조건(`Weather_Condition`) 기준 평균값으로 대체
  - Rainy, Snowy, Foggy 날씨별로 다른 평균값 적용

#### 범주형
- 최빈값 또는 `unknown`으로 대체
- `Weather_Condition` 단순화: Clear, Cloudy, Rainy, Snowy, Foggy, Other로 그룹화

### 2-3 One-Hot Encoding
- `Wind_Direction` 단순화: North, South, East, West, Calm, Variable, Other로 그룹화
- `Weather_Condition` 등 범주형 변수 OHE 적용
- 편향된 Boolean 변수 제거 (`Amenity`, `Bump` 등 - 대부분 False)

### 2-4 Feature Scaling
- 스케일러 비교: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- 스케일링 전후 평균, 표준편차, 최소/최대값 비교 및 시각화
- 왜도가 큰 특성과 스케일이 큰 특성 자동 식별

### 2-5 Model Evaluation
- **StratifiedKFold (k=5)**를 사용한 K-fold 교차검증 (분류)
- **KFold (k=5)** 사용 (회귀)
- 클래스 불균형을 고려하여 stratified 방식 사용

---

## 3. Modeling & Evaluation

### Classification - 사고 심각도 예측
- RandomForestClassifier (class_weight='balanced')
- GradientBoostingClassifier
- XGBoostClassifier (scale_pos_weight 설정, AUCPR 평가지표 사용)
- ROC 곡선 및 Precision-Recall 곡선 분석

### Regression - 사고 처리 시간(Duration) 예측
- LinearRegression (log1p 변환 적용)
- RandomForestRegressor (GridSearch를 통한 하이퍼파라미터 최적화)
- XGBoostRegressor
- Feature Importance 분석 및 Top 10 특성 추출

## 성능 평가 지표

|     문제 유형    |                  사용 지표                     |
|----------------|---------------------------------------------|
| 분류 (Severity) | Accuracy, Weighted F1 Score (클래스 불균형 고려) |
| 회귀 (Duration) | RMSE, MAE, R² Score                         |

※ 분류 문제는 **Stratified K-Fold Cross Validation**을 적용
※ XGBoost 분류 모델은 **AUCPR** 및 **scale_pos_weight** 사용

---

## 모델 성능 비교 및 Top 5 조합

### 분류 (Severity) - 교차검증 기반

| 순위  |     스케일러      |       모델        |    지표    |  점수   |
|------|----------------|------------------|-----------|--------|
| 1    | StandardScaler | XGBoost          | F1 Score  | 0.7991 |
| 2    | MinMaxScaler   | XGBoost          | F1 Score  | 0.7948 |
| 3    | RobustScaler   | XGBoost          | F1 Score  | 0.7917 |
| 4    | MinMaxScaler   | GradientBoosting | F1 Score  | 0.7864 |
| 5    | StandardScaler | GradientBoosting | F1 Score  | 0.7832 |

**최종 선정 모델**: `XGBoost + StandardScaler` (F1 Score = 0.7991)

### 회귀 (Duration_Minutes) - 교차검증 기반

| 순위  |      스케일러     |           모델             |  지표  |  점수   |
|------|----------------|---------------------------|-------|--------|
| 1    | RobustScaler   | XGBoostRegressor          |   R²  | 0.8536 |
| 2    | StandardScaler | RandomForestRegressor     |   R²  | 0.8478 |
| 3    | RobustScaler   | RandomForestRegressor     |   R²  | 0.8452 |
| 4    | MinMaxScaler   | XGBoostRegressor          |   R²  | 0.8410 |
| 5    | StandardScaler | GradientBoostingRegressor |   R²  | 0.8396 |

**최종 선정 모델**: `XGBoostRegressor + RobustScaler` (R² = 0.8536)

---

## 주요 개선사항 (하이퍼파라미터 및 베스트 모델 선정)

### 🔧 모델링 프로세스 개선
1. **교차검증 안정화**: 
   - 분류: StratifiedKFold로 클래스 불균형 고려
   - 회귀: KFold 5회 반복으로 성능 평균±표준편차 제공

2. **하이퍼파라미터 최적화**:
   - RandomForestRegressor: GridSearchCV 적용
   - XGBoost: scale_pos_weight로 클래스 불균형 처리
   - 최적 모델 자동 저장 (.joblib 형식)

3. **특성 중요도 분석**:
   - Top 10 중요 특성 추출 및 시각화
   - 모델별 Feature Importance 비교

4. **성능 지표 다각화**:
   - 분류: Accuracy, F1-Macro, F1-Weighted, AUCPR
   - 회귀: RMSE, MAE, R² Score

---

## 분석 해석 및 학습 경험

- **결측치 처리의 중요성**: 날씨별 Precipitation 처리 방식이 모델 성능에 큰 영향
- **클래스 불균형 대응**: StratifiedKFold와 scale_pos_weight로 효과적 해결
- **스케일링 효과**: RobustScaler가 이상치가 많은 데이터에서 우수한 성능
- **XGBoost 우수성**: 대부분 상황에서 최고 성능, 하이퍼파라미터 튜닝 효과 큼
- **협업 경험**: 모듈화, 함수 분할, 코드 리뷰를 통한 코드 품질 향상

### 주요 발견사항
- **중요 특성**: Start_Hour, Temperature, Visibility, Distance 등이 높은 중요도
- **날씨 영향**: 간소화된 날씨 조건이 원본보다 더 나은 예측 성능
- **시간 특성**: 시간대, 요일, 월별 특성이 사고 심각도와 처리시간에 큰 영향

---

## 오픈소스 기여

- **전처리 함수 모듈화**: `preprocessing.py`, `feature_scaling.py`
- **자동화된 결측치 처리**: 데이터 타입별 최적 전략 적용
- **교차검증 개선**: 클래스 불균형을 고려한 안정적 성능 평가
- **하이퍼파라미터 최적화**: GridSearch 기반 자동 튜닝
- **성능 비교 자동화**: 다양한 스케일러×모델 조합 자동 평가
- **완전한 문서화**: 주석, 사용 예시, 실행 가이드 포함
- **GitHub Repository**: [https://github.com/banchan316/Term_project_t11](https://github.com/banchan316/Term_project_t11)

### 재사용 가능한 함수들
```python
# 결측치 처리
X_imputed, missing_indices = handle_missing_value(X, verbose=True)

# 스케일링 및 시각화
X_scaled, scaler = scale_features(X, method='robust', verbose=True)
visualize_scaling_effect(X_original, X_scaled)

# 특성 분석
numeric_stats = analyze_numeric_features(X, visualize=True)

# 모델 학습 및 평가 (교차검증 포함)
model = severity_model_xgb(X, y, n_splits=5)
```

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

## 실행 방법

```bash
# 전체 파이프라인 실행
python main.py

# 개선된 교차검증 버전 실행  
python open_source.py

# 데이터 탐색
python inspection.py
```
