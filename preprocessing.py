import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    #원본 데이터

    print("1. 원본 데이터 정보 ") 
    print(f"데이터 형태: {df.shape}")
    df.info()
    print(df.head())

    #타겟 변수 생성 - 사고 심각도는 있으니 처리 시간만 생성(start_time, end_time은 버려도 되려나)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed') #뒤에 00000때메 mixed 사용 
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60 #초 단위 float으로 바꾸기 
 
    y_severity = df['Severity'].copy() #Severity 타겟
    y_duration = df['Duration_Minutes'].copy() #Duration 타겟

    #feature engineering 
    print("\n--- 3. 시간 관련 피처 생성 ---")
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek #요일 0 ~ 6정수로 바꾸기 

    features_to_drop = [
        'ID', 'Source', 'Start_Time', 'End_Time'
        'End_Lat', 'End_Lng', #결측치가 너무 많음
        'Description',
        'Timezone', 'Airport_Code', 'Weather_Timestamp', #불필요힌 정보
        'Wind_Chill(F)', # Temperature와 상관관계 높고 결측치 많음
        'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight' # Twilight 컬럼들 (Sunrise_Sunset과 유사하며, 카디널리티 줄이기 위해 하나만 남김)
    ] 



df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
preprocessing(df)

#print(df.head(5))
#print(df.shape)