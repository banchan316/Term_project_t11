import numpy as np
import pandas as pd
import feature_scaling as fs

def drop(df,target_name):

    """원본 데이터
    print("1. 원본 데이터 정보\n")
    print(f"데이터 형태: {df.shape}")
    df.info()
    print(df.head())"""

    #feature engineering

    #타겟 변수 생성 - 사고 심각도는 있으니 처리 시간만 생성(start_time, end_time은 버려도 되려나)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed') #뒤에 00000때메 mixed 사용
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60 #초 단위 float으로 바꾸기
    print("음수 Duration 개수:", (df['Duration_Minutes'] < 0).sum())    
    print("7일 초과 Duration 개수:", (df['Duration_Minutes'] > 1440 * 7).sum())
    df = df[df['Duration_Minutes'] >= 0] #음수 Duration 제거
    df = df[df['Duration_Minutes'] <= 1440 * 7]  #7일보다 긴 이상치 제거

    print("시간 관련 피처 생성")
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek #요일 0 ~ 6정수로 바꾸기

    #타겟 값 설정 
    if target_name == 'Severity':
        df['Severity_binary'] = np.where(df['Severity'] <= 2, 0, 1) #1,2 / 3,4로 묶기 
        y = df['Severity_binary'].astype(int).copy()
    elif target_name == 'Duration_Minutes':
        y = df['Duration_Minutes'].copy()#Duration_Minutes를 y로 사용
    else:
        print("에러요.")

    cat_names = [
    'Source', 'Severity', 'City', 'County', 'State', 'Country',
    'Timezone', 'Weather_Condition', 'Wind_Direction',
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
    ]

    for col in cat_names:
        print(col, df[col].unique().size)#country 단일 클래스라 삭제 

    features_to_drop = [
        'ID', 'Source', 'End_Time',
        'End_Lat', 'End_Lng', # inspection 결과 결측치가 너무 많음 
        'Distance(mi)', #사고 후에나 알 수 있음
        'Country', #단일 클래스 
        'Description', 'Weather_Timestamp', #불필요한 정보
        'Wind_Chill(F)', # Temperature와 상관관계 높고 결측치 많음
        'Street', 'City', 'Zipcode', 'Timezone', 'Airport_Code','County','State', #굳이 필요한가
        'Amenity', 'Bump', 'Give_way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Turning_Loop', #inspection 결과 너무 편향됨
        'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight' # Twilight 컬럼들 (Sunrise_Sunset과 유사하며, 카디널리티 줄이기 위해 하나만 남김)
    ]
    if target_name == 'Severity':
        features_to_drop.extend(['Severity','Severity_binary'])
    elif target_name == 'Duration_Minutes':
        features_to_drop.append('Duration_Minutes')

    df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    print(f"제거된 피처: {features_to_drop}")

    x = df.copy()

    return x,y


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
    # weather 값 단순화 
    df_processed['Weather_Condition'] = df_processed['Weather_Condition'].astype(str).str.lower().map(
        lambda val:
            'Foggy' if 'fog' in val or 'mist' in val or 'haze' in val else
            'Clear' if 'clear' in val or 'fair' in val else
            'Cloudy' if 'cloud' in val or 'overcast' in val else
            'Rainy' if 'rain' in val or 'drizzle' in val or 'shower' in val or 'thunder' in val or 't-storm' in val or 'wintry mix' in val else
            'Snowy' if 'snow' in val or 'sleet' in val or 'ice pellet' in val or 'freezing' in val else
            'Other'
    )

# 수치형 결측치 처리
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            missing_indices[col] = df_processed[df_processed[col].isnull()].index.tolist()

            if col == 'Precipitation(in)':
                rain_avg = df_processed[df_processed['Weather_Condition'] == 'Rainy'][col].mean()
                snow_avg = df_processed[df_processed['Weather_Condition'] == 'Snowy'][col].mean()
                foggy_avg = df_processed[df_processed['Weather_Condition'] == 'Foggy'][col].mean()
            
                fill_series = pd.Series(0.0, index=df_processed.index)
                fill_series[df_processed['Weather_Condition'] == 'Rainy'] = rain_avg
                fill_series[df_processed['Weather_Condition'] == 'Snowy'] = snow_avg
                fill_series[df_processed['Weather_Condition'] == 'Foggy'] = foggy_avg

                df_processed[col] = df_processed[col].fillna(fill_series)

                if verbose:
                    filled_count = len(missing_indices[col])
                    print(f"수치형 컬럼 'Precipitation(in)'의 결측치 {filled_count}개를 간소화된 날씨 기준 평균으로 대체했습니다.")
            else:
                if num_strategy == 'median':
                    value = df_processed[col].median()
                elif num_strategy == 'mean':
                    value = df_processed[col].mean()
                elif num_strategy == 'zero':
                    value = 0
                else:
                    value = df_processed[col].median()

            df_processed[col] = df_processed[col].fillna(value)

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

def one_hot_encoding(df):
    
    # 원본 데이터 복사
    df_encoded = df.copy()

    object_cols = df_encoded.select_dtypes(include=['object', 'category']).columns #범주형인 데이터만
    
    print('원본 범주형 데이터 클래스')
    for col in object_cols:
        unique_vals = df_encoded[col].unique()
        print(f"[{col}] ({len(unique_vals)}개 클래스):\n{unique_vals}\n")
        
    #Wind_Direction 값 단순화
    df_encoded['Wind_Direction'] = df_encoded['Wind_Direction'].map(
        lambda val: (
            'North' if val in ['N', 'North', 'NNW', 'NW', 'NE', 'NNE'] else
            'South' if val in ['S', 'South', 'SSW', 'SW', 'SE', 'SSE'] else
            'East' if val in ['E', 'East', 'ENE', 'ESE'] else
            'West' if val in ['W', 'West', 'WNW', 'WSW'] else
            'Calm' if val in ['Calm', 'CALM'] else
            'Variable' if val in ['VAR', 'Variable'] else
            'Other'
        )
    )
    
    print('단순화 후 클래스')
    for col in object_cols:
        unique_vals = df_encoded[col].unique()
        print(f"[{col}] ({len(unique_vals)}개 클래스):\n{unique_vals}\n")
        
    #onehotencoding
    df_encoded = pd.get_dummies(df_encoded, columns=object_cols)

    #결과 확인
    print('One_Hot_Encoding 후 shpae: ', df_encoded.shape)

    for col in object_cols:
        print(f"{col} One_Hot_Encoding 후 컬럼들:")
        for c in df_encoded.columns:
            if col in c:
                print(" -", c)
        print()
   
    return df_encoded

if __name__ == "__main__":
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    # column drop 및 타겟 ,변수 생성 
    x,y = drop(df, 'Severity') #Severity 또는 Duration_Minutes 입력 
    x.to_csv("US_Accidents_dropped.csv", index=False)

    #결측치 처리 
    X_imputed, missing_indices = handle_missing_value(x, verbose=True)
    print(f"결측치 처리 완료: {X_imputed.shape}")
    missing_count = X_imputed.isnull().sum().sum()
    print(f"남아있는 결측치 수: {missing_count}")
    
    #원 핫 인코딩 
    X_encoded = one_hot_encoding(X_imputed)
    print(f" One_Hot_Encoding 완료: {X_encoded.shape}")
    X_encoded.to_csv("US_Accidents_encoded.csv", index=False)
    
    #특성 분석  
    numeric_stats = fs.analyze_numeric_features(X_encoded, visualize = False )
    
    # 5. 정규화 (feature_scaling.py)
    print("\n5. 정규화(StandardScaler) 적용")
    X_scaled, scaler = fs.scale_features(X_encoded, method='standard', verbose=True)
    print(f"정규화 완료: {X_scaled.shape}")
    
    # 6. 다른 스케일링 방법 테스트 (MinMaxScaler)
    print("\n6. 정규화(MinMaxScaler) 적용")
    X_minmax, minmax_scaler = fs.scale_features(X_encoded, method='minmax', verbose=True)
    
    # 7. 다른 스케일링 방법 테스트 (RobustScaler)
    print("\n7. 정규화(RobustScaler) 적용")
    X_robust, robust_scaler = fs.scale_features(X_encoded, method='robust', verbose=True)

    fs.visualize_scaling_effect(X_encoded,X_scaled)
    fs.visualize_scaling_effect(X_encoded,X_minmax)
    fs.visualize_scaling_effect(X_encoded,X_robust)

    
    





