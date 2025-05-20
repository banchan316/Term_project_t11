import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute(df, missing_indices=None, n_neighbors=5, verbose=True):
    """
    df : pandas DataFrame (결측치 처리를 할 데이터프레임 (이미 전처리된 상태))
    missing_indices : dict, default=None (컬럼별 결측치가 있던 인덱스 정보 (선택적), None이면 df에서 결측치가 있는 행들을 직접 찾아 처리)
    n_neighbors : int, default=5 (KNN 알고리즘에서 사용할 이웃 개수)
    verbose : bool, default=True (처리 과정 출력 여부)
    df_imputed : pandas DataFrame (KNN으로 결측치가 처리된 데이터프레임)
    """
    # 원본 데이터 복사
    df_result = df.copy()
    
    # 날짜 타입 컬럼 제외 (KNN 적용 불가)
    date_cols = df_result.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols and verbose:
        print(f"날짜 타입 컬럼 {len(date_cols)}개는 KNN 처리에서 제외됩니다: {date_cols}")
    
    # 수치형 컬럼만 선택 (KNN 적용 가능)
    numeric_cols = df_result.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_cols:
        if verbose:
            print("KNN 처리할 수치형 컬럼이 없습니다.")
        return df_result
    
    if verbose:
        print(f"KNN 처리할 수치형 컬럼: {len(numeric_cols)}개")
    
    # 불리언 컬럼 처리 (KNNImputer는 숫자로 변환해야 함)
    bool_cols = df_result.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        if verbose:
            print(f"불리언 컬럼 {len(bool_cols)}개를 숫자로 변환합니다.")
        for col in bool_cols:
            df_result[col] = df_result[col].astype(int)
            numeric_cols.append(col)
    
    # 결측치가 있는지 확인
    has_missing = False
    for col in numeric_cols:
        if df_result[col].isnull().sum() > 0:
            has_missing = True
            break
    
    if not has_missing and missing_indices is None:
        if verbose:
            print("처리할 결측치가 없습니다.")
        return df_result
    
    # 결측치 처리용 데이터 준비 (수치형 열만)
    df_numeric = df_result[numeric_cols].copy()
    
    if missing_indices is not None:
        # missing_indices가 제공된 경우, 해당 인덱스에만 NaN 설정
        df_missing = df_numeric.copy()
        total_missing = 0
        
        for col, indices in missing_indices.items():
            if col in df_missing.columns:
                df_missing.loc[indices, col] = np.nan
                total_missing += len(indices)
        
        if verbose:
            print(f"KNN으로 처리할 결측치 수: {total_missing}")
    else:
        # missing_indices가 없으면 현재 df_numeric의 결측치 그대로 사용
        df_missing = df_numeric
        if verbose:
            missing_count = df_missing.isnull().sum().sum()
            print(f"KNN으로 처리할 결측치 수: {missing_count}")
    
    # 작은 데이터 세트에서만 실행 (대용량 데이터에서는 시간이 오래 걸릴 수 있음)
    if len(df_missing) > 100000 and n_neighbors > 10:
        if verbose:
            print(f"데이터 크기가 큽니다 ({len(df_missing)} 행). KNN의 이웃 수를 {n_neighbors}에서 5로 줄입니다.")
        n_neighbors = 5
    
    # KNN Imputer 적용
    if verbose:
        print(f"KNN Imputer(n_neighbors={n_neighbors})를 적용")
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_missing)
    
    # 결과를 데이터프레임으로 변환
    df_imputed = pd.DataFrame(imputed_array, columns=df_missing.columns, index=df_missing.index)
    
    # 처리 결과 반영
    if missing_indices is not None:
        # missing_indices가 제공된 경우, 해당 인덱스의 값만 업데이트
        for col, indices in missing_indices.items():
            if col in df_imputed.columns:
                df_result.loc[indices, col] = df_imputed.loc[indices, col].values
    else:
        # missing_indices가 없으면 모든 수치형 열 업데이트
        for col in numeric_cols:
            df_result[col] = df_imputed[col]
    
    # 불리언 컬럼 원래 타입으로 변환
    if bool_cols:
        for col in bool_cols:
            df_result[col] = df_result[col].astype(bool)
    
    if verbose:
        if missing_indices is not None:
            processed_count = sum(len(indices) for col, indices in missing_indices.items() 
                                 if col in numeric_cols)
            print(f"KNN 처리 완료: {processed_count}개 결측치 처리됨 (지정된 인덱스)")
        else:
            print(f"KNN 처리 완료: 모든 수치형 결측치 처리됨")
    
    return df_result


if __name__ == "__main__":
    # 테스트 코드
    import os
    from preprocessing import preprocessing
    from handle_missing_value import handle_missing_value
    from one_hot_encoding import one_hot_encoding
    from feature_scaling import scale_features

    # 1. 데이터 로드 & 전처리 파이프라인
    file_path = "US_Accidents_March23_sampled_500k.csv"
    df = pd.read_csv(file_path)
    
    # 2. 전처리 단계
    X, y = preprocessing(df, target_name='Severity')
    
    # 2.1 결측치 처리 및 인덱스 저장
    X_imputed, missing_indices = handle_missing_value(X, verbose=True)
    
    # 2.2 원-핫 인코딩
    X_encoded = one_hot_encoding(X_imputed)
    
    # 2.3 스케일링
    X_scaled, _ = scale_features(X_encoded, method='standard', verbose=True)
    
    # 3. KNN 결측치 처리 (수치형 열만)
    # 주의: 날짜 타입과 같은 비수치형 열은 제외됨
    X_knn = knn_impute(X_scaled, missing_indices=missing_indices, n_neighbors=5)
    
    # 4. 결과 저장
    os.makedirs('processed_data', exist_ok=True)
    X_knn.to_csv('processed_data/X_knn_imputed.csv', index=False)
    pd.Series(y).to_csv('processed_data/y.csv', index=False, header=['target'])
    print(f"KNN 결측치 처리 결과가 저장되었습니다: processed_data/X_knn_imputed.csv")