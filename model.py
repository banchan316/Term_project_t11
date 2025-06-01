from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from itertools import product
import joblib

def severity_model(X, y, n_splits=5, random_state=42):

    # KFold 정의
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracy_scores = []
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        accuracy_scores.append(acc)
        f1_scores.append(f1)

        print(f"[Fold {fold}] Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
        print(classification_report(y_val, y_pred, digits=4))

    print("\n 교차검증 평균 성능:")
    print(f"평균 Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"평균 F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    # 전체 데이터로 최종 모델 학습
    final_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    final_model.fit(X, y)

    y_pred = final_model.predict(X)
    acc_final = accuracy_score(y, y_pred)
    f1_final = f1_score(y, y_pred, average='macro')

    print("\n전체 데이터 학습 후 성능:")
    print(f"Accuracy (train on all): {acc_final:.4f}")
    print(f"F1 (macro): {f1_final:.4f}")
    print("\n 전체 데이터 분류 리포트:")
    print(classification_report(y, y_pred, digits=4))

    return final_model

def severity_model_xgb(X, y, n_splits=5, random_state=42):

    weight =  3 #402090 / 97498 처음 돌렸을 때도 recall이 너무 낮아서 추가함 -> 너무 precision이 낮아져서 3으로 조정 

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_list, f1_list = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective='binary:logistic',
            eval_metric='aucpr', # precision-recall 기반 평가
            use_label_encoder=False,
            scale_pos_weight= weight,
            random_state=random_state,
            verbosity=0
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)

        print(f"[Fold {fold}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_val, y_pred, digits=4))

    print("\n교차검증 평균 성능:")
    print(f"평균 Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"평균 F1 (macro): {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    # 전체 데이터로 최종 모델 훈련
    final_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='binary:logistic',
        eval_metric='aucpr',
        use_label_encoder=False,
        scale_pos_weight= weight,
        random_state=random_state,
        verbosity=0
    )
    final_model.fit(X, y)

    # 전체 데이터 예측 결과 확인
    y_pred_all = final_model.predict(X)
    acc_final = accuracy_score(y, y_pred_all)
    f1_final = f1_score(y, y_pred_all, average='macro')

    print("\전체 데이터 학습 후 성능:")
    print(f"Accuracy (train on all): {acc_final:.4f}")
    print(f"F1 (macro): {f1_final:.4f}")
    print("\n 전체 데이터 분류 리포트:")
    print(classification_report(y, y_pred_all, digits=4))

    return final_model

def duration_model_linear(X, y): #예측을 거의 못 하는데용

    y_log = np.log1p(y) #편차가 너무 커서

    model = LinearRegression()
    model.fit(X, y_log)

    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n 전체 데이터 평가 결과:")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE : {mae:.4f}")
    print(f" R²  : {r2:.4f}")

    return model

def duration_model_rf(X, y, random_state=42, save_path="best_rf_model.joblib"):
    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    keys = list(param_grid.keys())
    best_score = float('inf')
    best_params = None
    best_model = None

    print("하이퍼파라미터 조합 탐색 시작...\n")

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))

        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params
        )

        model.fit(X, y)
        y_pred = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"→ {params} | RMSE: {rmse:.4f}")

        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = model

    # 최적 모델로 최종 평가
    y_pred = best_model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n 최적 하이퍼파라미터:")
    print(best_params)
    print("\n 전체 데이터 평가 결과:")
    print(f"RMSE: {best_score:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    # 최적 모델 저장
    joblib.dump(best_model, save_path)
    print(f"\n모델이 '{save_path}'에 저장되었습니다.")

    return best_model

def duration_model_rf_top10_log(X_full, y, top_features, random_state=42):
    X_selected = X_full[top_features]
    y_log = np.log1p(y)

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    keys = list(param_grid.keys())
    best_score = float('inf')
    best_params = None
    best_model = None

    print("하이퍼파라미터 조합 탐색 시작...\n")

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))

        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params
        )

        model.fit(X_selected, y_log)
        y_log_pred = model.predict(X_selected)
        y_pred = np.expm1(y_log_pred)

        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"→ {params} | RMSE: {rmse:.4f}")

        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = model

    print("\n최적 하이퍼파라미터:")
    print(best_params)

    # 최종 성능 평가
    y_log_pred = best_model.predict(X_selected)
    y_pred = np.expm1(y_log_pred)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n최종 모델 성능 (Train 기준):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return best_model
    

if __name__ == "__main__":
    df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")

    