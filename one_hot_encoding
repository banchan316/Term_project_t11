import pandas as pd

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
    
    #Weather_Condition 값 단순화
    df_encoded['Weather_Condition'] = df_encoded['Weather_Condition'].astype(str).str.lower()

    df_encoded['Weather_Condition'] = df_encoded['Weather_Condition'].map(lambda val:
        'Clear' if 'clear' in val or 'fair' in val else
        'Partly Cloudy' if 'partly cloudy' in val or 'scattered' in val or 'mostly cloudy' in val else
        'Cloudy' if 'cloudy' in val or 'overcast' in val else
        'Rain' if 'rain' in val or 'drizzle' in val or 'shower' in val else
        'Thunderstorm' if 'thunder' in val or 't-storm' in val else
        'Snow' if 'snow' in val or 'sleet' in val or 'ice pellet' in val or 'freezing' in val else
        'Fog/Mist' if 'fog' in val or 'mist' in val or 'haze' in val or 'smoke' in val else
        'Wintry Mix' if 'wintry mix' in val else
        'Dust/Sand' if 'dust' in val or 'sand' in val else
        'Windy' if 'windy' in val else
        'Severe' if 'tornado' in val or 'funnel' in val or 'volcanic' in val else
        'Other'
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
    # 테스트 코드
    try:
        import os
        df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
        
        try:
            from preprocessing import preprocessing
            from handle_missing_value import handle_missing_value
            
            X, y = preprocessing(df, 'Severity')
            
            # 결측치 처리
            X_imputed = handle_missing_value(X, verbose=True)
            
            # 처리 결과 확인
            print(f"결측치 처리 완료: {X_imputed.shape}")

            #인코딩
            X_encoded = one_hot_encoding(X_imputed)

            # 처리 결과 확인
            print(f" One_Hot_Encoding 완료: {X_encoded.shape}")

        except ImportError:
            print("임포트 실패.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
