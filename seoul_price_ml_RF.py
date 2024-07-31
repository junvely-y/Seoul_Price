import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 불러오기
train_file_path = 'seoul_apt_price_lat_lon_add_train_with_subway_info.csv'
test_file_path = 'seoul_apt_price_lat_lon_add_test_with_subway_info.csv'

logging.info(f"Loading train data from {train_file_path}")
train_df = pd.read_csv(train_file_path)

logging.info(f"Loading test data from {test_file_path}")
test_df = pd.read_csv(test_file_path)

# 필요 없는 열 제거
columns_to_drop = ['도로명', '시군구', '매수자', '매도자']
logging.info(f"Dropping columns: {columns_to_drop}")
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# 계약년월을 datetime 형식으로 변환하고 연도와 월로 분리
logging.info("Converting 계약년월 to datetime and extracting year and month")
train_df['계약년월'] = pd.to_datetime(train_df['계약년월'], format='%Y%m')
test_df['계약년월'] = pd.to_datetime(test_df['계약년월'], format='%Y%m')

train_df['계약연도'] = train_df['계약년월'].dt.year
train_df['계약월'] = train_df['계약년월'].dt.month
test_df['계약연도'] = test_df['계약년월'].dt.year
test_df['계약월'] = test_df['계약년월'].dt.month

# 계약년월 컬럼 삭제
train_df = train_df.drop(columns=['계약년월'])
test_df = test_df.drop(columns=['계약년월'])


# 결측값 처리 (간단한 예로 결측값을 제거하는 방법을 사용)
logging.info("Dropping rows with NaN values")
train_df = train_df.dropna()
test_df = test_df.dropna()

# 타겟 변수와 피처 분리
logging.info("Splitting target and features")
X_train = train_df.drop(columns=['거래금액(만원)'])
y_train = train_df['거래금액(만원)']

X_test = test_df.drop(columns=['거래금액(만원)'])
y_test = test_df['거래금액(만원)']

# 범주형 변수와 수치형 변수 분리
categorical_features = ['단지명', '지하철역명', '거래유형', '지역구']
numerical_features = [col for col in X_train.columns if col not in categorical_features]
logging.info(f"Categorical features: {categorical_features}")
logging.info(f"Numerical features: {numerical_features}")

# 전처리 파이프라인
logging.info("Creating preprocessing pipeline")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 훈련 데이터와 검증 데이터로 분할
logging.info("Splitting train data into training and validation sets")
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 전처리 파이프라인을 훈련 데이터에 맞게 학습 (fit)
logging.info("Fitting preprocessing pipeline on training data")
preprocessor.fit(X_train_split)

# 훈련 데이터, 검증 데이터, 테스트 데이터에 대해 전처리 파이프라인을 적용 (transform)
logging.info("Transforming training, validation, and test data")
X_train_split_transformed = preprocessor.transform(X_train_split)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# 모델 정의 및 학습
logging.info("Creating model pipeline")
model = RandomForestRegressor(n_estimators=100, random_state=42)

logging.info("Training the model")
model.fit(X_train_split_transformed, y_train_split)

# 검증 데이터로 예측
logging.info("Predicting with the model on validation set")
y_val_pred = model.predict(X_val_transformed)

# 평가 (검증 데이터)
logging.info("Evaluating the model on validation set")
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)

logging.info(f'Validation Mean Absolute Error: {val_mae}')
logging.info(f'Validation Mean Squared Error: {val_mse}')
logging.info(f'Validation Root Mean Squared Error: {val_rmse}')

# 테스트 데이터로 예측
logging.info("Predicting with the model on test set")
y_test_pred = model.predict(X_test_transformed)

# 평가 (테스트 데이터)
logging.info("Evaluating the model on test set")
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

logging.info(f'Test Mean Absolute Error: {test_mae}')
logging.info(f'Test Mean Squared Error: {test_mse}')
logging.info(f'Test Root Mean Squared Error: {test_rmse}')

# 결과 출력
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})

logging.info("Saving results to 'predictions.csv'")
results.to_csv('predictions.csv', index=False)

print(f'Validation Mean Absolute Error: {val_mae}')
print(f'Validation Mean Squared Error: {val_mse}')
print(f'Validation Root Mean Squared Error: {val_rmse}')

print(f'Test Mean Absolute Error: {test_mae}')
print(f'Test Mean Squared Error: {test_mse}')
print(f'Test Root Mean Squared Error: {test_rmse}')
