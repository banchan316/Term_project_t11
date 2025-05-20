import numpy as np
import pandas as pd

def handle_missing_value(df, num_strategy='median', cat_strategy='most_frequent', verbose=True):
    """
    df : pandas DataFrame (처리할 데이터프레임)
    num_strategy : str, default='median' (수치형 결측치 대체 - 중앙값)
    cat_strategy : str, default='most_frequent' (범주형 결측치 대체 - 최빈값)
    verbose : bool, default=True (과정과 결과를 출력할지 여부)
    df_processed : pandas DataFrame (결측치가 처리된 데이터프레임)
    missing_indices : dict (각 컬럼별 결측치가 있던 인덱스 정보)
    """
    # 원본 데이터 복사
    df_processed = df.copy()
    
    # 결측치 인덱스를 저장할 딕셔너리
    missing_indices = {}
    
    # 현재 결측치 현황 확인
    missing_count_before = df_processed.isnull().sum().sum()
    if verbose:
        print(f"\n처리 전 결측치 수: {missing_count_before}")
    
    # 수치형/범주형 컬럼 구분
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    # 수치형 컬럼 처리
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            # 결측치 인덱스 저장
            missing_indices[col] = df_processed[df_processed[col].isnull()].index.tolist()
            
            if num_strategy == 'median':
                value = df_processed[col].median()
            elif num_strategy == 'mean':
                value = df_processed[col].mean()
            elif num_strategy == 'zero':
                value = 0
            else:
                value = df_processed[col].median()  # 기본값은 중앙값
            
            df_processed.loc[:, col] = df_processed[col].fillna(value)
            
            if verbose:
                print(f"수치형 컬럼 '{col}'의 결측치 {len(missing_indices[col])}개를 {num_strategy}({value:.2f})로 대체했습니다.")
    
    # 범주형 컬럼 처리
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            # 결측치 인덱스 저장
            missing_indices[col] = df_processed[df_processed[col].isnull()].index.tolist()
            
            if cat_strategy == 'most_frequent':
                value = df_processed[col].mode()[0]
            elif cat_strategy == 'constant':
                value = 'unknown'
            else:
                value = df_processed[col].mode()[0]  # 기본값은 최빈값
            
            df_processed.loc[:, col] = df_processed[col].fillna(value)
            
            if verbose:
                print(f"범주형 컬럼 '{col}'의 결측치 {len(missing_indices[col])}개를 {cat_strategy}('{value}')로 대체했습니다.")
    
    # 결과 확인
    missing_count_after = df_processed.isnull().sum().sum()
    if verbose:
        print(f"처리 후 결측치 수: {missing_count_after}")
        print(f"총 {missing_count_before - missing_count_after}개의 결측치가 처리되었습니다.")
        
        # 저장된 결측치 인덱스 정보 요약
        print(f"\n결측치 인덱스 정보 저장 완료: {len(missing_indices)}개 컬럼")
    
    return df_processed, missing_indices

if __name__ == "__main__":
    # 테스트 코드
    try:
        import os
        file_path = "US_Accidents_March23_sampled_500k.csv"
        df = pd.read_csv(file_path)
        
        # preprocessing.py의 preprocessing 함수 실행
        try:
            from preprocessing import preprocessing
            X, y = preprocessing(df, 'Severity')
            print(f"preprocessing.py의 기본 전처리 완료: {X.shape}")
            
            # 결측치 처리
            X_imputed, missing_indices = handle_missing_value(X, verbose=True)
            
            # 결측치 인덱스 정보 저장
            os.makedirs('processed_data', exist_ok=True)
            
            # JSON으로 저장 (인덱스만 저장)
            import json
            with open('processed_data/missing_indices.json', 'w') as f:
                # 인덱스는 정수 리스트로 저장
                json_compatible = {col: [int(idx) for idx in indices] for col, indices in missing_indices.items()}
                json.dump(json_compatible, f)
            
            # 처리된 데이터 저장
            X_imputed.to_csv('processed_data/data_imputed.csv', index=False)
            
            # 처리 결과 확인
            print(f"결측치 처리 완료: {X_imputed.shape}")
            print(f"처리된 데이터와 결측치 인덱스 정보가 'processed_data' 디렉토리에 저장되었습니다.")

        except ImportError:
            print("preprocessing.py 임포트 실패. 원본 데이터에서 직접 결측치 처리.")
            df_imputed, missing_indices = handle_missing_value(df, verbose=True)
            
            # 결측치 인덱스 정보 저장
            os.makedirs('processed_data', exist_ok=True)
            
            # JSON으로 저장
            import json
            with open('processed_data/missing_indices_raw.json', 'w') as f:
                json_compatible = {col: [int(idx) for idx in indices] for col, indices in missing_indices.items()}
                json.dump(json_compatible, f)
            
            # 처리된 데이터 저장
            df_imputed.to_csv('processed_data/raw_data_imputed.csv', index=False)
            
            print(f"결측치 처리 완료: {df_imputed.shape}")
            print(f"처리된 데이터와 결측치 인덱스 정보가 'processed_data' 디렉토리에 저장되었습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")