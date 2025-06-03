import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

import feature_scaling as fs
import preprocessing as pp

def main():
    target_options = ['Severity', 'Duration_Minutes']

    for TARGET in target_options:
        print(f"==========================")
        print(f" 타겟: {TARGET}")
        print(f"==========================")

        df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
        x, y = pp.drop(df, TARGET)
        x.to_csv("US_Accidents_dropped.csv", index=False)

        # 결측치 처리 
        X_imputed, missing_indices = pp.handle_missing_value(x, verbose=True)
        print(f"결측치 처리 완료: {X_imputed.shape}") 

        # 원 핫 인코딩 
        X_encoded = pp.one_hot_encoding(X_imputed)
        print(f"One_Hot_Encoding 완료: {X_encoded.shape}")

        X = X_encoded

        # Scaling
        X_standard, _ = fs.scale_features(X, method='standard')
        X_minmax, _ = fs.scale_features(X, method='minmax')
        X_robust, _ = fs.scale_features(X, method='robust')

        scaled_sets = {
            "StandardScaler": X_standard,
            "MinMaxScaler": X_minmax,
            "RobustScaler": X_robust
        }

        # 분류 or 회귀
        if TARGET == 'Severity':
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            scorers = {
                "accuracy": accuracy_score,
                "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
            }
            sort_descending = True

            cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # StratifiedKFold 사용 (클래스 불균형 고려)

        else:  # Duration_Minutes
            models = {
                "LinearRegression": LinearRegression(),
                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoostRegressor": XGBRegressor(n_estimators=100, random_state=42)
            }
            scorers = {
                "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                "R2": r2_score
            }
            sort_descending = False

            cv_method = KFold(n_splits=5, shuffle=True, random_state=42) # KFold 사용

        results = []

        for scaler_name, X_scaled in scaled_sets.items():
            print(f"\n [{scaler_name}] 정규화 적용")
            
            # 교차검증으로 더 안정적인 평가
            cv_results = []
            for train_idx, test_idx in cv_method.split(X_scaled, y):
                X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    for score_name, scorer in scorers.items():
                        score = scorer(y_test, preds)
                        cv_results.append((scaler_name, model_name, score_name, score))
            
            # 교차검증 평균 계산
            cv_df = pd.DataFrame(cv_results, columns=['scaler', 'model', 'metric', 'score'])
            cv_summary = cv_df.groupby(['scaler', 'model', 'metric'])['score'].agg(['mean', 'std']).reset_index()
            
            for _, row in cv_summary.iterrows():
                print(f"→ {row['model']} | {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")
                results.append((TARGET, row['scaler'], row['model'], row['metric'], row['mean']))

        # Top 5 출력
        print(f"\n[TOP 5 조합 - 타겟: {TARGET}]")
        
        # 주요 지표로 정렬 (분류: f1, 회귀: R2)
        if TARGET == 'Severity':
            main_metric_results = [r for r in results if r[3] == 'f1']
        else:
            main_metric_results = [r for r in results if r[3] == 'R2']
        
        results_sorted = sorted(main_metric_results, key=lambda x: x[4], reverse=sort_descending)
        for i, (tgt, scaler, model, metric, score) in enumerate(results_sorted[:5], 1):
            print(f"{i}. [{scaler}] {model} ({metric}) → {score:.4f}")

if __name__ == '__main__':
    main()
