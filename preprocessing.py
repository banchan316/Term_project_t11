import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(df,target_name):

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
        'ID', 'Source', 'Start_Time', 'End_Time',
        'End_Lat', 'End_Lng', # inspection 결과 결측치가 너무 많음 
        'Distance(mi)', #사고 후에나 알 수 있음
        'Country', #단일 클래스 
        'Description', 'Weather_Timestamp', #불필요한 정보
        'Wind_Chill(F)', # Temperature와 상관관계 높고 결측치 많음
        'Street', 'City', 'State', 'Zipcode', 'Timezone', 'Airport_Code', #굳이 필요한가
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
    
if __name__ == "__main__":
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    x,y = preprocessing(df, 'Severity') #Severity 또는 Duration_Minutes 입력 
    print(x.columns)  
    print(y.value_counts())

    

