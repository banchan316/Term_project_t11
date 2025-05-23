import numpy as np
import pandas as pd
import preprocessing as pp 
import feature_scaling as fs

def main():
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    # column drop 및 타겟 ,변수 생성 
    x,y = pp.drop(df, 'Severity') #Severity 또는 Duration_Minutes 입력 
    x.to_csv("US_Accidents_dropped.csv", index=False)

    #결측치 처리 
    X_imputed, missing_indices = pp.handle_missing_value(x, verbose=True)
    print(f"결측치 처리 완료: {X_imputed.shape}")
    missing_count = X_imputed.isnull().sum().sum()
    print(f"남아있는 결측치 수: {missing_count}")
    
    #원 핫 인코딩 
    X_encoded = pp.one_hot_encoding(X_imputed)
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
    return X_imputed, y

if __name__ == "__main__":
    main()