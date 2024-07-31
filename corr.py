import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 불러오기
file_path = 'seoul_apt_price_lat_lon_add_train_with_subway_info.csv'
logging.info(f"Loading data from {file_path}")
df = pd.read_csv(file_path, encoding='utf-8')

# 필요 없는 열 제거
columns_to_drop = ['도로명', '시군구', '매수자', '매도자']
df = df.drop(columns=columns_to_drop)

# 결측값 처리
df = df.dropna()

# 상관관계 행렬 계산
corr_matrix = df.corr()

# 상관관계 행렬 출력
logging.info("Correlation matrix:")
logging.info(f"\n{corr_matrix}")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 사용할 경우
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS에서 사용할 경우

# 상관관계 행렬 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간의 상관관계 행렬')
plt.show()
