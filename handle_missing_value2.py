import numpy as np
import pandas as pd

def handle_missing_value(df, date_col='Start_Time', location_col='State', 
                         num_strategy='median', cat_strategy='mode', 
                         verbose=True):
    """
    df : pandas DataFrame (처리할 데이터프레임)
    date_col : str, default='Start_Time' (날짜/시간 컬럼명)
    location_col : str, default='State' (위치 정보 컬럼명)
    num_strategy : str, default='median' (수치형 결측치 대체 전략 ('median', 'mean'))
    cat_strategy : str, default='mode' (범주형 결측치 대체 전략 ('mode'))
    verbose : bool, default=True (과정과 결과를 출력할지 여부)
    df_processed : pandas DataFrame (결측치가 처리된 데이터프레임)
    imputation_indices : dict (각 처리 단계별로 처리된 인덱스 정보)
    """
    # 원본 데이터 복사
    df_processed = df.copy()
    
    # 결측치 인덱스 저장용 딕셔너리
    imputation_indices = {
        'date_location': {}, # 1차: 날짜+위치 기반
        'date_only': {},     # 2차: 날짜만 기반
        'global': {}         # 3차: 전체 데이터 기반
    }
    
    # 결측치 현황 확인
    if verbose:
        missing_before = df_processed.isnull().sum().sum()
        print(f"처리 전 결측치 수: {missing_before}")
    
    # 날짜 컬럼 처리
    if date_col in df_processed.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_processed[date_col]):
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
        df_processed['date_only'] = df_processed[date_col].dt.date
    else:
        print(f"'{date_col}' 컬럼이 없습니다.")
        return df_processed, imputation_indices
    
    # 위치 컬럼 확인
    if location_col not in df_processed.columns:
        location_col = None
        if verbose:
            print(f"'{location_col}' 컬럼이 없어 날짜만으로 처리합니다.")
    
    # 수치형/범주형 컬럼 구분
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 처리 대상에서 제외할 컬럼들
    exclude_cols = ['date_only', date_col]
    if location_col:
        exclude_cols.append(location_col)
    
    # 날짜/위치 컬럼 제외
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # 결측치가 있는 컬럼만 처리
    numeric_cols_with_missing = [col for col in numeric_cols if df_processed[col].isnull().sum() > 0]
    categorical_cols_with_missing = [col for col in categorical_cols if df_processed[col].isnull().sum() > 0]
    
    # 수치형 컬럼 처리
    for col in numeric_cols_with_missing:
        # 인덱스 저장 초기화
        for method in imputation_indices:
            imputation_indices[method][col] = []
        
        # 결측치 인덱스 저장
        null_indices = df_processed[df_processed[col].isnull()].index.tolist()
        
        # 1단계: 날짜 + 위치 기반 대체
        if location_col:
            # 그룹별 통계량 계산
            if num_strategy == 'median':
                date_loc_values = df_processed.groupby(['date_only', location_col])[col].transform('median')
            else:  # 'mean'
                date_loc_values = df_processed.groupby(['date_only', location_col])[col].transform('mean')
            
            # 결측치 대체 전후 인덱스 비교
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            df_processed.loc[df_processed[col].isnull(), col] = date_loc_values[df_processed[col].isnull()]
            after_indices = df_processed[df_processed[col].isnull()].index.tolist()
            
            # 1단계에서 처리된 인덱스 저장
            filled_indices = list(set(before_indices) - set(after_indices))
            imputation_indices['date_location'][col] = filled_indices
            
            if verbose and filled_indices:
                print(f"'{col}': 1단계(날짜+위치)에서 {len(filled_indices)}개 처리")
        
        # 2단계: 날짜 기반 대체
        if df_processed[col].isnull().sum() > 0:
            # 그룹별 통계량 계산
            if num_strategy == 'median':
                date_values = df_processed.groupby('date_only')[col].transform('median')
            else:  # 'mean'
                date_values = df_processed.groupby('date_only')[col].transform('mean')
            
            # 결측치 대체 전후 인덱스 비교
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            df_processed.loc[df_processed[col].isnull(), col] = date_values[df_processed[col].isnull()]
            after_indices = df_processed[df_processed[col].isnull()].index.tolist()
            
            # 2단계에서 처리된 인덱스 저장
            filled_indices = list(set(before_indices) - set(after_indices))
            imputation_indices['date_only'][col] = filled_indices
            
            if verbose and filled_indices:
                print(f"'{col}': 2단계(날짜)에서 {len(filled_indices)}개 처리")
        
        # 3단계: 전체 데이터 기반 대체
        if df_processed[col].isnull().sum() > 0:
            # 전체 데이터 통계량 계산
            if num_strategy == 'median':
                global_value = df_processed[col].median()
            else:  # 'mean'
                global_value = df_processed[col].mean()
            
            # 결측치 대체 전 인덱스 저장
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            df_processed.loc[df_processed[col].isnull(), col] = global_value
            
            # 3단계에서 처리된 인덱스 저장
            imputation_indices['global'][col] = before_indices
            
            if verbose and before_indices:
                print(f"'{col}': 3단계(전체)에서 {len(before_indices)}개 처리")
    
    # 범주형 컬럼 처리
    for col in categorical_cols_with_missing:
        # 인덱스 저장 초기화
        for method in imputation_indices:
            imputation_indices[method][col] = []
        
        # 1단계: 날짜 + 위치 기반 대체
        if location_col:
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            
            for (date, loc), group in df_processed.groupby(['date_only', location_col]):
                # 그룹에 유효한 값이 있는지 확인
                valid_values = group[col].dropna()
                if len(valid_values) > 0:  # 유효한 값이 하나 이상 있는 경우만 처리
                    mode_value = valid_values.mode().iloc[0]  # 최빈값 계산
                    mask = ((df_processed['date_only'] == date) & 
                            (df_processed[location_col] == loc) & 
                            df_processed[col].isnull())
                    df_processed.loc[mask, col] = mode_value
            
            # 처리된 인덱스 계산
            after_indices = df_processed[df_processed[col].isnull()].index.tolist()
            filled_indices = list(set(before_indices) - set(after_indices))
            imputation_indices['date_location'][col] = filled_indices
            
            if verbose and filled_indices:
                print(f"'{col}': 1단계(날짜+위치)에서 {len(filled_indices)}개 처리")
        
        # 2단계: 날짜 기반 대체
        if df_processed[col].isnull().sum() > 0:
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            
            for date, group in df_processed.groupby('date_only'):
                valid_values = group[col].dropna()
                if len(valid_values) > 0:  # 유효한 값이 하나 이상 있는 경우만 처리
                    mode_value = valid_values.mode().iloc[0]  # 최빈값 계산
                    mask = (df_processed['date_only'] == date) & df_processed[col].isnull()
                    df_processed.loc[mask, col] = mode_value
            
            # 처리된 인덱스 계산
            after_indices = df_processed[df_processed[col].isnull()].index.tolist()
            filled_indices = list(set(before_indices) - set(after_indices))
            imputation_indices['date_only'][col] = filled_indices
            
            if verbose and filled_indices:
                print(f"'{col}': 2단계(날짜)에서 {len(filled_indices)}개 처리")
        
        # 3단계: 전체 데이터 기반 대체
        if df_processed[col].isnull().sum() > 0:
            before_indices = df_processed[df_processed[col].isnull()].index.tolist()
            
            valid_values = df_processed[col].dropna()
            if len(valid_values) > 0:  # 유효한 값이 하나 이상 있는 경우만 처리
                mode_value = valid_values.mode().iloc[0]  # 최빈값 계산
                df_processed.loc[df_processed[col].isnull(), col] = mode_value
                
                # 3단계에서 처리된 인덱스는 남은 모든 결측치
                imputation_indices['global'][col] = before_indices
                
                if verbose and before_indices:
                    print(f"'{col}': 3단계(전체)에서 {len(before_indices)}개 처리")

    # 임시 컬럼 제거
    if 'date_only' in df_processed.columns:
        df_processed = df_processed.drop('date_only', axis=1)
    
    # 결과 확인
    if verbose:
        missing_after = df_processed.isnull().sum().sum()
        print(f"\n처리 결과: {missing_before - missing_after}개 결측치 처리됨")
        
        # 처리 단계별 통계
        dl_count = sum(len(indices) for col_indices in imputation_indices['date_location'].values() for indices in [col_indices])
        do_count = sum(len(indices) for col_indices in imputation_indices['date_only'].values() for indices in [col_indices])
        gl_count = sum(len(indices) for col_indices in imputation_indices['global'].values() for indices in [col_indices])
        
        print(f"1단계(날짜+위치): {dl_count}개")
        print(f"2단계(날짜): {do_count}개")
        print(f"3단계(전체): {gl_count}개")
    
    return df_processed, imputation_indices

if __name__ == "__main__":
    # 테스트 코드
    try:
        import os
        file_path = "US_Accidents_March23_sampled_500k.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # preprocessing.py 사용
            try:
                from preprocessing import preprocessing
                X, y = preprocessing(df, 'Severity')
                print(f"전처리 완료: {X.shape}")
                
                # 결측치 처리
                X_imputed, imp_indices = handle_missing_value(
                    X, date_col='Start_Time', location_col='State', verbose=True
                )
                
                # 결과 확인
                missing = X_imputed.isnull().sum().sum()
                print(f"남은 결측치: {missing}개")
                
            except ImportError:
                print("preprocessing.py 임포트 실패.")
                
        else:
            print(f"파일 없음: {file_path}")
    
    except Exception as e:
        print(f"오류: {e}")