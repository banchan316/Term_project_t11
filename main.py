import numpy as np
import pandas as pd
from preprocessing import preprocessing
from handle_missing_value import handle_missing_value

def main():
    # 데이터 로드
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    print(f"데이터 로드 완료: {df.shape}")
    
    # 기본 전처리 (preprocessing.py 활용)
    X, y = preprocessing(df, target_name='Severity')
    print(f"기본 전처리 완료: {X.shape}")
    
    # 결측치 처리 (handle_missing_value.py 활용)
    X_imputed = handle_missing_value(X, verbose=True)
    print(f"결측치 처리 완료: {X_imputed.shape}")
    
    # 처리된 데이터 저장
    X_imputed.to_csv("US_Accidents_missing_values_handled.csv", index=False)
    print("처리된 데이터를 'US_Accidents_missing_values_handled.csv'로 저장")
    
    return X_imputed, y

if __name__ == "__main__":
    main()