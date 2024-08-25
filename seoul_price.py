import pandas as pd
import os
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("geocoding.log", encoding='utf-8'), 
                              logging.StreamHandler()])

# 데이터 파일 경로 설정
train_file_path = 'seoul_apt_price_train_1y.csv'
test_file_path = 'seoul_apt_price_test.csv'
train_output_file = 'seoul_apt_price_lat_lon_add_train.csv'
test_output_file = 'seoul_apt_price_lat_lon_add_test.csv'

# api_key = '카카오맵 API 키'

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='cp949')
    logging.info(f"Data loaded from {file_path}. Total rows: {len(df)}")

    columns_needed = ['단지명', '전용면적(㎡)', '계약년월', '계약일', '거래금액(만원)', 
                      '층', '매수자', '매도자', '건축년도', '도로명', '거래유형', '해제사유발생일', '시군구']

    df = df[columns_needed]
    df = df[df['해제사유발생일'] == '-']
    df = df.drop(columns=['해제사유발생일'])
    logging.info(f"Necessary columns extracted. Remaining rows: {len(df)}")

    df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)
    logging.info("Transaction amounts cleaned and converted to integers")

    return df

def get_lat_lon(address):
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'documents' in data and len(data['documents']) > 0:
            lat = data['documents'][0]['y']
            lon = data['documents'][0]['x']
            return lat, lon
        else:
            return None, None
    else:
        return None, None

def add_lat_lon(df_road):
    latitudes = []
    longitudes = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_lat_lon, row['도로명']): row['도로명'] for _, row in df_road.iterrows()}
        
        for future in as_completed(futures):
            address = futures[future]
            lat, lon = future.result()
            latitudes.append(lat)
            longitudes.append(lon)
            logging.info(f"Processed {len(latitudes)} / {len(df_road)}")
    
    df_road['위도'] = latitudes
    df_road['경도'] = longitudes
    return df_road

def process_and_save_data(df, output_file):
    df_road = df[['도로명']].drop_duplicates().reset_index(drop=True)
    logging.info(f"Unique road addresses extracted. Total unique addresses: {len(df_road)}")

    df_road = add_lat_lon(df_road)

    df['위도'] = df['도로명'].map(df_road.set_index('도로명')['위도'])
    df['경도'] = df['도로명'].map(df_road.set_index('도로명')['경도'])

    # 고층 판단: 층수가 15층 이상인 경우 고층으로 판단
    df['고층'] = df['층'].apply(lambda x: 1 if x >= 15 else 0)

    # 신축 여부: 건축년도가 최근 5년 이내인 경우 신축으로 판단
    current_year = pd.to_datetime('today').year
    def categorize_building_year(year):
        age = current_year - year
        if age <= 5:
            return 1
        elif 6 <= age <= 10:
            return 2
        else:
            return 3

    df['신축'] = df['건축년도'].apply(categorize_building_year)
    
    # 지역구 추출: 시군구에서 지역구 추출
    df['지역구'] = df['시군구'].apply(lambda x: x.split()[1])

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Geocoded data saved to {output_file}")

# 학습 데이터 전처리
train_df = load_and_preprocess_data(train_file_path)
process_and_save_data(train_df, train_output_file)

# 테스트 데이터 전처리
test_df = load_and_preprocess_data(test_file_path)
process_and_save_data(test_df, test_output_file)
