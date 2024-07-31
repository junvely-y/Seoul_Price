import folium
import pandas as pd
import requests
import json

# 파일 경로 설정
train_file_path = 'seoul_apt_price_lat_lon_add_train_with_subway_info.csv'
subway_file_path = 'tnSubwayStatn_with_coordinates.csv'

# 데이터 읽기
train_df = pd.read_csv(train_file_path)
subway_df = pd.read_csv(subway_file_path)

# NaN 값 제거
train_df = train_df.dropna(subset=['위도', '경도'])
subway_df = subway_df.dropna(subset=['Latitude', 'Longitude'])

# 서울 GeoJSON 데이터 직접 받아오기
r = requests.get('https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json')
seoul_geo = r.json()

# 서울 지도 생성
m = folium.Map(location=[37.52408, 126.9802], zoom_start=11, tiles='cartodbpositron')

# 서울 GeoJSON 경계 추가
folium.GeoJson(seoul_geo, name='지역구').add_to(m)

# 아파트 위치 추가
for idx, row in train_df.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"{row['단지명']}<br>Price: {row['거래금액(만원)']}"
    ).add_to(m)

# 지하철역 위치 추가
for idx, row in subway_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"Station: {row['지하철역명']}"
    ).add_to(m)

# 지도 저장
m.save('seoul_apartments_subway_stations.html')
