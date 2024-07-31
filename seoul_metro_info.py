import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CSV 파일 읽기
file_path = 'tnSubwayStatn.csv'
df = pd.read_csv(file_path)

# 지하철역명에서 '응암순환'을 '응암'으로 변경
df['지하철역명'] = df['지하철역명'].apply(lambda x: x.replace('응암순환', '응암'))
df['지하철역명'] = df['지하철역명'].apply(lambda x: x.replace('송정', '서울송정'))

# 특정 지하철역 제거
df = df[~df['지하철역명'].str.contains('지평', na=False)]
df = df[~df['지하철역명'].str.contains('양평', na=False)]

# 서울이 아닌 지역이 포함된 행 제거
keywords = ['인천광역시', '경기도', '충청북도', '경기', '강원도', '충청남도', '인천시']
df = df[~df['기본주소'].str.contains('|'.join(keywords), na=False)]

# 지하철역명에서 '(' 이후의 모든 내용 제거
df['지하철역명'] = df['지하철역명'].str.split('(').str[0] + '역'

# 필요한 컬럼만 선택
need_columns = ['지하철역명']
df_filtered = df[need_columns]

# Kakao API 설정
KAKAO_API_KEY = 'f49e4c6c00131a2a854e2b63747ee78a'
KAKAO_MAP_URL = 'https://dapi.kakao.com/v2/local/search/keyword.json'

def get_station_coordinates(station_name):
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": station_name}
    response = requests.get(KAKAO_MAP_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['documents']:
            location = data['documents'][0]
            return station_name, location['y'], location['x']
    return station_name, None, None

# 기존의 환승노선수가 정확하지 않으며, 지하철역명이 여러번 나오게 되는데 이를 활용하여 환승노선수 작성
transfer_counts = df_filtered['지하철역명'].value_counts().to_dict()
df_filtered['노선개수'] = df_filtered['지하철역명'].map(transfer_counts)

df_filtered = df_filtered.drop_duplicates(subset=['지하철역명'])

# 병렬 처리 설정
num_workers = 10  # 병렬로 실행할 스레드 수
results = []

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(get_station_coordinates, station_name) for station_name in df_filtered['지하철역명']]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            logging.info(f'Processed {result[0]} - Latitude: {result[1]}, Longitude: {result[2]}')
        except Exception as e:
            logging.error(f'Error processing station: {e}')

# 위도와 경도를 저장할 리스트
station_names = []
latitudes = []
longitudes = []

for station_name, lat, lon in results:
    station_names.append(station_name)
    latitudes.append(lat)
    longitudes.append(lon)

# 데이터프레임에 위도와 경도 컬럼 추가
df_filtered['Latitude'] = latitudes
df_filtered['Longitude'] = longitudes

df_filtered['지하철역명'] = df_filtered['지하철역명'].apply(lambda x: x.replace('서울송정', '송정'))

# 결과 저장
output_file_path = 'tnSubwayStatn_with_coordinates.csv'
df_filtered.to_csv(output_file_path, index=False, encoding='utf-8-sig')

logging.info(f'CSV file with coordinates saved as {output_file_path}')
