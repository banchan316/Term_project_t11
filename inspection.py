import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import plotly.express as px

#파일 불러오기
df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")

# 1. Check distribution of major categorical variables

print("[1] Distribution of major categorical variable values (Top 10):")
categorical_cols = df.select_dtypes(include='object').columns.tolist()
top_cats = ['State', 'Weather_Condition', 'Wind_Direction', 'City', 'Timezone']

for col in top_cats:
    print(f"\n Value distribution in {col}:")
    print(df[col].value_counts(dropna=False).head(10))
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, y=col, hue=col, order=df[col].value_counts().head(10).index, palette="Set2", legend=False)
    plt.title(f"Top 10 Value Distribution in {col}")
    plt.tight_layout()
    plt.show()
# 2. 심각도 분포 확인 

print("\n[2] Distribution of Severity class:")
severity_counts = df['Severity'].value_counts()
severity_ratio = severity_counts / len(df)
print(severity_ratio, severity_counts)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Severity', hue='Severity', palette='pastel', legend=False)
plt.title("Distribution of Severity Class")
plt.tight_layout()
plt.show()

# 3. 미싱 데이터 비율이 높은 변수 
# (Missing values count)
print("\n Missing Values Count:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

print("\n[3] Variables with missing value ratio ≥ 40%:")
null_ratio = df.isnull().mean().sort_values(ascending=False)
high_nulls = null_ratio[null_ratio > 0.4]
print(high_nulls)

# missing data 시각화 
plt.figure(figsize=(14,5))
sns.barplot(x=null_ratio[null_ratio > 0].index, y=null_ratio[null_ratio > 0].values)
plt.title("Missing Ratio of All Variables with Missing Values")
plt.ylabel("Missing Ratio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#4. 중복 row 확인 
dup_count = df.duplicated().sum()
print(f"\n[4] Number of duplicated rows: {dup_count}")

bool_features = [
    col for col in df.columns 
    if set(df[col].dropna().unique()).issubset({True, False}) 
    # dropna() : 결측값 때문에 고유값이 {True, False} 집합 외의 다른 값을 포함할 수 있기 때문
    # set() : 집합 -> 중복된 요소 없으므로, 값들의 종류 명확하게 확인가능
]

# 각 feature에 대해 true, false, missing 값의 비율(%) 출력
for feature in bool_features:
    total = len(df[feature])
    true_count = np.sum(df[feature] == True)
    false_count = np.sum(df[feature] == False)
    missing_count = df[feature].isnull().sum()
    
    print(f"{feature}:")
    print(f"  True    : {true_count / total * 100:.2f}%")
    print(f"  False   : {false_count / total * 100:.2f}%")
    print(f"  Missing : {missing_count / total * 100:.2f}%\n")

# Inspection 5. Boolean 변수의 값 분포 확인 (with pie chart)
# boolean feature 지정
bool_features = [col for col in df.columns 
                 if set(df[col].dropna().unique()).issubset({True, False})]

# 서브플롯 grid 생성
num_features = len(bool_features)
num_cols = 4
num_rows = (num_features + num_cols - 1) // num_cols  # 올림 연산

fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
axs = axs.flatten()  # 1차원 배열로 변환하여 쉽게 순회

for i, col in enumerate(bool_features):
    data = df[col]
    total = len(data)
    true_count = data.eq(True).sum()
    false_count = data.eq(False).sum()
    missing_count = data.isnull().sum()
    
    # 분포 데이터: True와 False만 있는 경우
    sizes = [true_count, false_count]
    labels = ['True', 'False']
    colors = ['#0078FF', '#E54C4C']
    
    # 결측치가 있는 경우, 추가
    if missing_count > 0:
        sizes.append(missing_count)
        labels.append('Missing')
        colors.append('gray')
    
    axs[i].pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, 
               startangle=90, counterclock=False)
    axs[i].set_title(col)

# 서브플롯 수보다 남는 축들은 숨기기
for j in range(i+1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95) # 위에 짤리는거 수정 
plt.show()

print(df[['Start_Lat', 'Start_Lng', 'Severity']].head())
df['Severity'] = df['Severity'].astype(str)
# Plotly Express의 scatter_mapbox를 이용한 US 지도 산점도 생성
fig = px.scatter_mapbox(
    df,
    lat="Start_Lat",             
    lon="Start_Lng",             
    color="Severity",            
    opacity = 0.6,
    color_discrete_sequence = ["green", "yellow","orange","red"],
    size_max=10,                 
    zoom=3,                      # 지도 확대율 (미국 전체가 보이도록)
    mapbox_style="open-street-map",
    title="US Accidents: Severity Scatter Plot"
)

fig.show()

