import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import severity_model, severity_model_xgb


def run_evaluation():
    """
    모델 평가 함수
     전처리된 'US_Accidents_encoded.csv' 파일을 로드하여 사고 심각도(Severity) 예측 모델(RF, XGBoost)을 평가
    """

    # 데이터 불러오기
    df = pd.read_csv('US_Accidents_encoded.csv')

    # 타겟과 특성 분리
    X = df.drop(columns=['Severity'])  # 특성
    y = df['Severity']                 # 타겟

    # y 값이 0부터 시작하도록 변환 (XGBoost가 요구함)
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y))  # [1,2,3,4] -> [0,1,2,3]

    print("===== Random Forest 모델 평가 =====")
    rf_model = severity_model(X, y_encoded)

    print("\n\n===== XGBoost 모델 평가 =====")
    xgb_model = severity_model_xgb(X, y_encoded)

if __name__ == '__main__':
    run_evaluation()

