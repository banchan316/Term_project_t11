import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    #원본 데이터
    print("1. 원본 데이터 정보\n")
    #print(f"데이터 형태: {df.shape}")
    #df.info()
    #print(df.head())

    #feature engineering

    #타겟 변수 생성 - 사고 심각도는 있으니 처리 시간만 생성(start_time, end_time은 버려도 되려나)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed') #뒤에 00000때메 mixed 사용
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60 #초 단위 float으로 바꾸기
    df = df[df['Duration_Minutes'] >= 0] # 음수 Duration 제거

    y_severity = df['Severity'].copy() #Severity 타겟
    y_duration = df['Duration_Minutes'].copy() #Duration 타겟

    print("시간 관련 피처 생성")
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek #요일 0 ~ 6정수로 바꾸기

    cat_names = ['Country','Timezone','Amenity','Bump','Crossing',
                 'Give_Way','Junction','No_Exit','Railway','Roundabout',
                 'Station','Stop','Traffic_Calming','Traffic_Signal',
                 'Turning_Loop','Sunrise_Sunset','Civil_Twilight',
                 'Nautical_Twilight','Astronomical_Twilight']

    for col in cat_names:
        print(col, df[col].unique().size) #county랑 turning_loop 단일 클래스라 삭제 

    features_to_drop = [
        'ID', 'Source', 'Start_Time', 'End_Time',
        'End_Lat', 'End_Lng','Distance(mi)', #사고 후에나 알 수 있음
        'County','Turning_Loop', #단일 클래스 
        'Description', 'Weather_Timestamp', #불필요한 정보
        'Wind_Chill(F)', # Temperature와 상관관계 높고 결측치 많음
        'Street', 'City', 'State', 'Zipcode', 'Timezone', 'Airport_Code', #굳이 필요한가
        'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight' # Twilight 컬럼들 (Sunrise_Sunset과 유사하며, 카디널리티 줄이기 위해 하나만 남김)
    ]

if __name__ == "__main__":
    
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    preprocessing(df)