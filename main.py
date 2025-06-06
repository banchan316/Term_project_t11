import numpy as np
import pandas as pd
import preprocessing as pp 
import feature_scaling as fs
import model as md  
import joblib
import matplotlib.pyplot as plt
def main():
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
    # column drop 및 타겟 ,변수 생성 
    x,y = pp.drop(df, 'Severity') #Severity 또는 Duration_Minutes 입력 
    #x,y = pp.drop(df, 'Duration_Minutes')
    x.to_csv("US_Accidents_dropped.csv", index=False)

    #결측치 처리 
    X_imputed, missing_indices = pp.handle_missing_value(x, verbose=True)
    print(f"결측치 처리 완료: {X_imputed.shape}")
    
    #원 핫 인코딩 
    X_encoded = pp.one_hot_encoding(X_imputed)
    print(f" One_Hot_Encoding 완료: {X_encoded.shape}")
    X_encoded.to_csv("US_Accidents_encoded.csv", index=False)
    
    # Scaling 뭐로 할지 정하기 
    """
    #특성 분석 
    numeric_stats = fs.analyze_numeric_features(X_encoded, visualize = False )
    
    print("\n5. 정규화(StandardScaler) 적용")
    X_scaled, scaler = fs.scale_features(X_encoded, method='standard', verbose=True)
    print(f"정규화 완료: {X_scaled.shape}")
    
    print("\n6. 정규화(MinMaxScaler) 적용")
    X_minmax, minmax_scaler = fs.scale_features(X_encoded, method='minmax', verbose=True)
    
    fs.visualize_scaling_effect(X_encoded,X_scaled)
    fs.visualize_scaling_effect(X_encoded,X_minmax)
    fs.visualize_scaling_effect(X_encoded,X_robust)
    """
    print("\n정규화(RobustScaler) 적용")
    X_robust, robust_scaler = fs.scale_features(X_encoded, method='robust', verbose=True)

    #모델 사용하고 싶은 거 하나씩 풀어서 보기 
    #model = md.severity_model(X_robust, y, n_splits = 5) #recall이 너무 낮음 -> 데이터가 불균형해 -> xgboost 사용해보기
    #model = md.severity_model_xgb(X_robust, y, n_splits = 5) 

    #model_duration = md.duration_model_linear(X_robust, y)
    #model_duration = md.duration_model_rf(X_robust, y)

    
    dt_model = joblib.load('best_rf_model.joblib')
    importances = dt_model.feature_importances_
    feature_names = X_encoded.columns
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False)
    print(feat_imp)
    
    top_n = 10
    top_features = feat_imp.head(top_n)
    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind='barh')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.grid(axis='x')
    plt.show()

    # model_top10 = md.duration_model_rf_top10_log(X_robust, y, top_features.index) - 10개 뽑은 거 
    
if __name__ == "__main__":
    main()