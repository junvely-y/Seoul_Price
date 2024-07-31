import pandas as pd
import numpy as np

# 파일 경로 설정
train_file_path = 'seoul_apt_price_lat_lon_add_train.csv'
test_file_path = 'seoul_apt_price_lat_lon_add_test.csv'
subway_file_path = 'tnSubwayStatn_with_coordinates.csv'

# 데이터 읽기
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
subway_df = pd.read_csv(subway_file_path)

# 지하철 데이터프레임 열 이름 확인
print(subway_df.columns)

# Euclidean distance를 계산하여 가장 가까운 지하철역을 찾는 함수
def find_closest_station(lat, lon, subway_df):
    distances = np.sqrt((subway_df['Latitude'] - lat)**2 + (subway_df['Longitude'] - lon)**2)
    min_index = distances.idxmin()
    closest_station = subway_df.loc[min_index, '지하철역명']  # '지하철역명' -> 'StationName' 수정
    num_lines = subway_df.loc[min_index, '노선개수']  # '노선개수' -> 'LineCount' 수정
    return closest_station, num_lines

# 각 train, test 데이터에 지하철역명과 노선개수 추가
def add_station_info(df, subway_df):
    station_names = []
    line_counts = []
    
    for idx, row in df.iterrows():
        lat = row['위도']
        lon = row['경도']
        if pd.notna(lat) and pd.notna(lon):  # NaN 값 처리
            station_name, line_count = find_closest_station(lat, lon, subway_df)
            station_names.append(station_name)
            line_counts.append(line_count)
        else:
            station_names.append(None)
            line_counts.append(None)
    
    df['지하철역명'] = station_names
    df['노선개수'] = line_counts

add_station_info(train_df, subway_df)
add_station_info(test_df, subway_df)

# 결과 파일 경로 설정
train_output_path = 'seoul_apt_price_lat_lon_add_train_with_subway_info.csv'
test_output_path = 'seoul_apt_price_lat_lon_add_test_with_subway_info.csv'

# 데이터 프레임을 CSV 파일로 저장
train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
test_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')

print("Train and test data with subway info have been saved successfully.")
