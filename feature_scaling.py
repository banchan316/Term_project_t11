import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
import os
from preprocessing import preprocessing
from handle_missing_value import handle_missing_value

# 수치형 특성의 분포 및 스케일 분석
def analyze_numeric_features(df, verbose=True, visualize=False):
    """
    df : pandas DataFrame (분석할 데이터프레임)
    verbose : bool, default=True (상세 정보를 출력할지 여부)
    visualize : bool, default=False (분포를 시각화할지 여부)
    numeric_stats : pandas DataFrame (수치형 특성의 통계 정보)
    """

    # 수치형 특성 선택
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        if verbose:
            print("수치형 특성이 없습니다.")
        return pd.DataFrame()
    
    # 기본 통계 계산
    numeric_stats = df[numeric_cols].describe().T
    
    # 추가 통계량(왜도, 첨도) 계산
    numeric_stats['skew'] = df[numeric_cols].skew()
    numeric_stats['kurtosis'] = df[numeric_cols].kurtosis()
    
    # 스케일 범위 정보 추가
    numeric_stats['range'] = numeric_stats['max'] - numeric_stats['min']
    
    if verbose:
        print(f"총 {len(numeric_cols)}개 수치형 특성 분석")
        
        # 스케일 차이가 큰 특성 찾기
        large_scale = numeric_stats[numeric_stats['range'] > 100].index.tolist()
        if large_scale:
            print(f"\n스케일이 큰 특성 ({len(large_scale)}개):")
            for col in large_scale:
                print(f"  - {col}: 범위 {numeric_stats.loc[col, 'range']:.2f}, 평균 {numeric_stats.loc[col, 'mean']:.2f}, 표준편차 {numeric_stats.loc[col, 'std']:.2f}")
        
        # 왜도가 큰 특성 찾기
        skewed = numeric_stats[numeric_stats['skew'].abs() > 1].index.tolist()
        if skewed:
            print(f"\n왜도가 큰 특성 ({len(skewed)}개):")
            for col in skewed:
                print(f"  - {col}: 왜도 {numeric_stats.loc[col, 'skew']:.2f}")
        
        print("\n수치형 특성 기본 통계:")
        print(numeric_stats)
    
    # 시각화
    if visualize:
        # 모든 수치형 특성 시각화
        plot_cols = numeric_cols  # 모든 수치형 특성 포함
        
        n_cols = 3  # 한 행에 3개 그래프
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols  # 행 수 (올림)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))

        # axes를 항상 2D 배열로 처리하기 위한 변환
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 모든 수치형 특성에 대해 히스토그램과 박스플롯 생성
        for i, col in enumerate(plot_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols

            # 히스토그램
            ax = axes[row_idx, col_idx]
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} distribution")
            ax.set_xlabel(col)
            
            # 보조 축에 박스플롯 추가
            ax2 = ax.twinx()
            sns.boxplot(x=df[col].dropna(), ax=ax2, color="lightgreen", orient="h", width=0.3)
            ax2.set_yticklabels([])
            
            # 통계 텍스트 표시
            stats_text = (
                f"mean: {df[col].mean():.2f}\n"
                f"std: {df[col].std():.2f}\n"
                f"min: {df[col].min():.2f}\n"
                f"max: {df[col].max():.2f}\n"
                f"skew: {df[col].skew():.2f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 남은 축 숨기기
        for i in range(len(plot_cols), n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 특성 간 스케일 비교 시각화
        plt.figure(figsize=(12, 6))
        sns.barplot(x=numeric_stats.index, y='range', data=numeric_stats)
        plt.title('Distribution of Numeric Features')
        plt.xticks(rotation=90)
        plt.ylabel('Range')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.show()
    
    return numeric_stats

# 수치형 특성에 일관된 스케일링 방법 적용
def scale_features(df, method='standard', cols=None, verbose=True):
    """
    df : pandas DataFrame (스케일링할 데이터프레임)
    method : str, default='standard' (스케일링 방법 ('standard', 'minmax', 'robust'))
    cols : list, default=None (스케일링할 컬럼 목록 (None이면 모든 수치형 특성))
    verbose : bool, default=True (과정과 결과를 출력할지 여부)
    df_scaled : pandas DataFrame (스케일링된 데이터프레임)
    scaler : object (학습된 스케일러 객체)
    """
    # 원본 데이터 복사
    df_scaled = df.copy()
    
    # 스케일링할 컬럼 선택
    if cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        numeric_cols = [col for col in cols if col in df.columns]
    
    if not numeric_cols:
        if verbose:
            print("스케일링할 수치형 특성이 없습니다.")
        return df_scaled, None
    
    # 스케일링 전 통계
    pre_stats = df[numeric_cols].describe().T
    
    # 결측치 확인 (결측치가 있으면 경고)
    missing_cols = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    if missing_cols:
        if verbose:
            print(f"특성에 결측치가 있습니다. 스케일링 전에 결측치를 처리하세요.")
            for col in missing_cols:
                print(f"  - {col}: {df[col].isnull().sum()}개 결측치")
        # 결측치가 있는 특성은 제외
        numeric_cols = [col for col in numeric_cols if col not in missing_cols]
        if not numeric_cols:
            if verbose:
                print("스케일링할 수치형 특성이 남아있지 않습니다.")
            return df_scaled, None
    
    # 스케일러 선택
    if method == 'standard':
        scaler = StandardScaler()
        if verbose:
            print(f"StandardScaler 적용: 평균 0, 표준편차 1로 변환")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        if verbose:
            print(f"MinMaxScaler 적용: 최소 0, 최대 1로 변환")
    elif method == 'robust':
        scaler = RobustScaler()
        if verbose:
            print(f"RobustScaler 적용: 중앙값 0, IQR로 스케일링")
    else:
        if verbose:
            print(f"{method}, 기본값인 StandardScaler 적용")
        scaler = StandardScaler()
    
    # 스케일링 적용
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 스케일링 후 통계
    post_stats = df_scaled[numeric_cols].describe().T
    
    if verbose:
        print(f"{len(numeric_cols)}개 수치형 특성에 {method} 스케일링 적용 완료")
        
        print("\n=== 스케일링 전후 비교 ===")
        comparison = pd.DataFrame({
            '변환 전 평균': pre_stats['mean'],
            '변환 전 표준편차': pre_stats['std'],
            '변환 전 최소값': pre_stats['min'],
            '변환 전 최대값': pre_stats['max'],
            '변환 후 평균': post_stats['mean'],
            '변환 후 표준편차': post_stats['std'],
            '변환 후 최소값': post_stats['min'],
            '변환 후 최대값': post_stats['max']
        })
        
        print(comparison)
        
        # StandardScaler의 경우 평균, 표준편차 검증
        if method == 'standard':
            print("\nStandardScaler 검증:")
            print(f"평균 기대값: 0.0")
            print(f"표준편차 기대값: 1.0")
            print(f"평균 실제값 (평균): {post_stats['mean'].mean():.6f}")
            print(f"표준편차 실제값 (평균): {post_stats['std'].mean():.6f}")
        
        # MinMaxScaler의 경우 최소, 최대값 검증
        elif method == 'minmax':
            print("\nMinMaxScaler 검증:")
            print(f"최소값 기대값: 0.0")
            print(f"최대값 기대값: 1.0")
            print(f"최소값 실제값 (평균): {post_stats['min'].mean():.6f}")
            print(f"최대값 실제값 (평균): {post_stats['max'].mean():.6f}")

        # RobustScaler의 경우 중앙값, IQR 검증
        elif method == 'robust':
            print("\nRobustScaler 검증:")
            print(f"중앙값 기대값: 0.0")
            print(f"중앙값 실제값 (평균): {post_stats['50%'].mean():.6f}")
            # 사분위수 범위 표시
            print(f"1사분위수 실제값 (평균): {post_stats['25%'].mean():.6f}")
            print(f"3사분위수 실제값 (평균): {post_stats['75%'].mean():.6f}")
    
    # 스케일러 객체 저장
    os.makedirs('models', exist_ok=True) # models 디렉토리 생성
    joblib.dump(scaler, f'models/{method}_scaler.joblib') # scaler 객체를 파일로 저장
    
    if verbose: # 스케일러 저장 성공 시 메시지 출력
        print(f"\n스케일러 객체를 'models/{method}_scaler.joblib'에 저장했습니다.")
    
    return df_scaled, scaler

# 이미 학습된 스케일러를 새 데이터에 적용
def apply_scaling(df, scaler, cols=None, verbose=True):
    """
    df : pandas DataFrame (스케일링할 데이터프레임)
    scaler : object (학습된 스케일러 객체)
    cols : list, default=None (스케일링할 컬럼 목록 (None이면 스케일러가 학습된 모든 특성))
    verbose : bool, default=True (과정과 결과를 출력할지 여부)
    df_scaled : pandas DataFrame (스케일링된 데이터프레임)
    """
    if scaler is None:
        if verbose:
            print("스케일러가 제공되지 않았습니다. 원본 데이터를 반환합니다.")
        return df.copy()
    
    # 원본 데이터 복사
    df_scaled = df.copy()
    
    # 학습된 특성명 확인 (가능한 경우)
    try:
        # StandardScaler, MinMaxScaler, RobustScaler는 feature_names_in_ 속성을 가짐
        scaler_features = scaler.feature_names_in_
        has_feature_names = True
    except:
        # 특성명을 가지지 않는 경우 (수동으로 적용된 스케일러 등)
        has_feature_names = False
    
    # 스케일링할 컬럼 선택
    if cols is None:
        if has_feature_names:
            # 스케일러가 학습된 특성만 선택
            numeric_cols = [col for col in scaler_features if col in df.columns]
        else:
            # 모든 수치형 특성 선택
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        numeric_cols = [col for col in cols if col in df.columns]
    
    if not numeric_cols:
        if verbose:
            print("스케일링할 수치형 특성이 없습니다.")
        return df_scaled
    
    # 결측치 확인 (결측치가 있으면 경고)
    missing_cols = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    if missing_cols:
        if verbose:
            print(f"특성에 결측치가 있습니다. 스케일링 전에 결측치를 처리하세요.")
            for col in missing_cols:
                print(f"  - {col}: {df[col].isnull().sum()}개 결측치")
        # 결측치가 있는 특성은 제외
        numeric_cols = [col for col in numeric_cols if col not in missing_cols]
        if not numeric_cols:
            if verbose:
                print("스케일링할 수치형 특성이 남아있지 않습니다.")
            return df_scaled
    
    # 스케일링 적용
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
    
    if verbose:
        print(f"{len(numeric_cols)}개 수치형 특성에 스케일링 적용")
        
        # 스케일러 유형 식별
        scaler_type = type(scaler).__name__
        
        # 스케일링 후 통계
        post_stats = df_scaled[numeric_cols].describe().T
        
        print("\n=== 스케일링 결과 통계 ===")
        print(post_stats)
        
        # 스케일러 유형에 따른 검증
        if scaler_type == 'StandardScaler':
            print("\nStandardScaler 검증:")
            print(f"평균 기대값: 0.0")
            print(f"표준편차 기대값: 1.0")
            print(f"평균 실제값 (평균): {post_stats['mean'].mean():.6f}")
            print(f"표준편차 실제값 (평균): {post_stats['std'].mean():.6f}")
        elif scaler_type == 'MinMaxScaler':
            print("\nMinMaxScaler 검증:")
            print(f"최소값 기대값: 0.0")
            print(f"최대값 기대값: 1.0")
            print(f"최소값 실제값 (평균): {post_stats['min'].mean():.6f}")
            print(f"최대값 실제값 (평균): {post_stats['max'].mean():.6f}")
        elif scaler_type == 'RobustScaler':
            print("\nRobustScaler 검증:")
            print(f"중앙값 기대값: 0.0")
            print(f"중앙값 실제값 (평균): {post_stats['50%'].mean():.6f}")
            print(f"1사분위수 실제값 (평균): {post_stats['25%'].mean():.6f}")
            print(f"3사분위수 실제값 (평균): {post_stats['75%'].mean():.6f}")
    
    return df_scaled

# 스케일링 전후 데이터 분포 시각화해서 비교
def visualize_scaling_effect(df_original, df_scaled, cols=None, max_cols=5):
    """
    df_original : pandas DataFrame (원본 데이터프레임)
    df_scaled : pandas DataFrame (스케일링된 데이터프레임)
    cols : list, default=None (시각화할 컬럼 목록 (None이면 자동 선택))
    max_cols : int, default=5 (최대 시각화할 컬럼 수)
    """
    # 시각화할 컬럼 선택
    if cols is None:
        # 공통 수치형 특성 찾기
        numeric_cols_original = df_original.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols_scaled = df_scaled.select_dtypes(include=['int64', 'float64']).columns
        cols = [col for col in numeric_cols_original if col in numeric_cols_scaled]
        
        # 최대 컬럼 수 제한
        cols = cols[:min(max_cols, len(cols))]
    
    if not cols:
        print("시각화할 공통 수치형 특성이 없습니다.")
        return
    
    # 행과 열 계산
    n_cols = 2  # 열 수 (원본, 스케일링 후)
    n_rows = len(cols)  # 행 수 (특성별)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    # 단일 행인 경우 axes 배열 형태 조정
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(cols):
        # 원본 데이터 분포
        sns.histplot(df_original[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"{col} - Before")
        
        # 스케일링된 데이터 분포
        sns.histplot(df_scaled[col], kde=True, ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} - After")
        
        # 통계 정보 추가
        original_stats = (
            f"mean: {df_original[col].mean():.2f}\n"
            f"std: {df_original[col].std():.2f}\n"
            f"min: {df_original[col].min():.2f}\n"
            f"max: {df_original[col].max():.2f}\n"
            f"range: {df_original[col].max() - df_original[col].min():.2f}"
        )
        
        scaled_stats = (
            f"mean: {df_scaled[col].mean():.2f}\n"
            f"std: {df_scaled[col].std():.2f}\n"
            f"min: {df_scaled[col].min():.2f}\n"
            f"max: {df_scaled[col].max():.2f}\n"
            f"range: {df_scaled[col].max() - df_scaled[col].min():.2f}"
        )

        # 왼쪽 상단에 통계 정보 표시 (원본)
        axes[i, 0].text(0.05, 0.95, original_stats, transform=axes[i, 0].transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 왼쪽 상단에 통계 정보 표시 (스케일링 후)
        axes[i, 1].text(0.05, 0.95, scaled_stats, transform=axes[i, 1].transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 데이터 로드
    print("1. 데이터 로드")
    file_path = "US_Accidents_March23_sampled_500k.csv"
    if not os.path.exists(file_path):
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        exit()
    
    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {df.shape}")
    
    # 2. 기본 전처리 (preprocessing.py)
    print("\n2. 기본 전처리 실행")
    X, y = preprocessing(df, target_name='Severity')
    print(f"기본 전처리 완료: {X.shape}")
    
    # 3. 결측치 처리 (handle_missing_value.py)
    print("\n3. 결측치 처리")
    X_imputed = handle_missing_value(X, verbose=True)
    print(f"결측치 처리 완료: {X_imputed.shape}")
    
    # 결측치 처리된 데이터의 결측치 확인
    missing_count = X_imputed.isnull().sum().sum()
    print(f"남아있는 결측치 수: {missing_count}")
    
    # 4. 수치형 특성 분석 (시각화 없이)
    print("\n4. 수치형 특성 분석")
    numeric_stats = analyze_numeric_features(X_imputed, visualize=False)
    
    # 5. 정규화 (feature_scaling.py)
    print("\n5. 정규화(StandardScaler) 적용")
    X_scaled, scaler = scale_features(X_imputed, method='standard', verbose=True)
    print(f"정규화 완료: {X_scaled.shape}")
    
    # 6. 다른 스케일링 방법 테스트 (MinMaxScaler)
    print("\n6. 정규화(MinMaxScaler) 적용")
    X_minmax, minmax_scaler = scale_features(X_imputed, method='minmax', verbose=True)
    
    # 7. 다른 스케일링 방법 테스트 (RobustScaler)
    print("\n7. 정규화(RobustScaler) 적용")
    X_robust, robust_scaler = scale_features(X_imputed, method='robust', verbose=True)
    
    # 8. 저장된 스케일러로 새 데이터 테스트
    print("\n8. 저장된 스케일러로 새 데이터 테스트")
    
    # 테스트용 샘플 데이터 생성 (원본 데이터의 일부)
    sample_size = min(1000, len(X_imputed))
    X_sample = X_imputed.sample(sample_size, random_state=42)
    
    # 각 스케일러로 샘플 데이터 변환
    print("\n8.1. StandardScaler 적용 테스트:")
    X_sample_std = apply_scaling(X_sample, scaler, verbose=True)
    
    print("\n8.2. MinMaxScaler 적용 테스트:")
    X_sample_minmax = apply_scaling(X_sample, minmax_scaler, verbose=True)
    
    print("\n8.3. RobustScaler 적용 테스트:")
    X_sample_robust = apply_scaling(X_sample, robust_scaler, verbose=True)
    
    # 9. 스케일링 결과 비교 (첫 번째 행 출력)
    print("\n9. 스케일링 결과 비교 (첫 번째 행):")
    compare_df = pd.DataFrame({
        'Original': X_sample.iloc[0],
        'StandardScaler': X_sample_std.iloc[0],
        'MinMaxScaler': X_sample_minmax.iloc[0],
        'RobustScaler': X_sample_robust.iloc[0]
    }).T
    
    print(compare_df)
    
    # 10. 결과 저장
    print("\n10. 정규화된 데이터 저장")
    os.makedirs('processed_data', exist_ok=True)
    X_scaled.to_csv('processed_data/X_standard_scaled.csv', index=False)
    X_minmax.to_csv('processed_data/X_minmax_scaled.csv', index=False)
    X_robust.to_csv('processed_data/X_robust_scaled.csv', index=False)
    pd.Series(y).to_csv('processed_data/y.csv', index=False, header=['target'])
    
    print("테스트 완료! 정규화된 데이터가 'processed_data' 폴더에 저장되었습니다.")