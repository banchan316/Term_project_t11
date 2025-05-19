import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
