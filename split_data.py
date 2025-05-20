#train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def default_split(X, y, train_size = 0.8, val_size = 0.1, save_csv=False):
    """
    X, y 데이터를 train/val/test로 분할

    X (pd.DataFrame): 입력데이터
    y (pd.Series): 정답값
    train_size (float): 학습 데이터 비율 (기본값 0.8)
    val_size (float): 전체 데이터에서 validation 비율 (기본값 0.1)
    save_csv (bool): CSV 파일 저장 유무
    """
  
    # 1 테스트 데이터 나누기 
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, train_size=train_size + val_size, random_state=0, stratify=y
    )
    
    # 2 검증셋 데이터 나누기
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=0, stratify=y_temp
    )

    #데이터 저장 
    if save_csv:
        X_train.to_csv(f"split data/X_train.csv", index=False)
        y_train.to_csv(f"split data/y_train.csv", index=False)
        X_val.to_csv(f"split data/X_val.csv", index=False)
        y_val.to_csv(f"split data/y_val.csv", index=False)    
        X_test.to_csv(f"split data/X_test.csv", index=False)
        y_test.to_csv(f"split data/y_test.csv", index=False)
        print('defalut split 저장 완료')


def kfold_split(X, y, n_splits=5, save_csv=False):
    """
    X, y 데이터를 K-Fold로 분할
    X (pd.DataFrame): 입력 데이터
    y (pd.Series): 타겟값
    n_splits (int): 폴드 수
    save_csv (bool): True

    모델 적용할 때
    for fold in range(1, 6):  # Fold 1~5
        X_train = pd.read_csv(f"kfold/X_train_fold{fold}.csv")
        y_train = pd.read_csv(f"kfold/y_train_fold{fold}.csv")
        X_val = pd.read_csv(f"kfold/X_val_fold{fold}.csv")
        y_val = pd.read_csv(f"kfold/y_val_fold{fold}.csv")

        학습 모델.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(x_val)
    """
    #Stratified K Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # 데이터 분할 
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # k = 5면 20개의 csv파일 저장 
        if save_csv:
            X_train.to_csv(f"kfold/X_train_fold{fold+1}.csv", index=False)
            y_train.to_csv(f"kfold/y_train_fold{fold+1}.csv", index=False)
            X_val.to_csv(f"kfold/X_val_fold{fold+1}.csv", index=False)
            y_val.to_csv(f"kfold/y_val_fold{fold+1}.csv", index=False)
            print(f"Fold {fold+1} 저장 완료")


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

            default_split(X, y)
            kfold_split(X, y)

            print("처리 완료")
        except ImportError:
            print("임포트 실패.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
    
