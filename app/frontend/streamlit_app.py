import streamlit as st
import pandas as pd
from zipfile import ZipFile 

import plotly.express as px
import asyncio
from request_utils import post_data

st.title("Team 61. Deepfake-Classification.")
st.header("Загрузка датасета")
st.text('Загрузите ZIP-архив с вашими данными для обучения модели. \nСтруктура каталога и имена директорий внутри архива должны соответствовать формату ниже:')
st.text("""
		.zip
		├──dataset
		├──┴─test
		├────┼─FAKE
		├────┴─REAL
		├──┴─train
		├────┼─FAKE
		└────┴─REAL""")

uploaded_file = st.file_uploader("Выберите ZIP-архив", type=['zip'])


if uploaded_file is not None:
	# Сохраняем zip-архив от пользователя в папку
	with open(r"..\data\load_user_dataset.zip", "wb") as f:
		f.write(uploaded_file.getbuffer())

	with ZipFile(r"..\data\load_user_dataset.zip", 'r') as zObject: 
  
    # Extracting all the members of the zip  
    # into a specific location. 
		zObject.extractall(path=r"..\data")
		st.write("ZIP-file saved and unziped")

if uploaded_file is not None:
	st.header(f"Анализ данных из {uploaded_file.name}")
	response_eda = asyncio.run(post_data('eda', input_data={}))

	df = pd.DataFrame(response_eda)
	df = df.T.copy()
	df['mean_red'] = df['mean_rgb'].apply(lambda x: x[0])
	df['mean_green'] = df['mean_rgb'].apply(lambda x: x[1])
	df['mean_blue'] = df['mean_rgb'].apply(lambda x: x[2])

	df['std_red'] = df['std_rgb'].apply(lambda x: x[0])
	df['std_green'] = df['std_rgb'].apply(lambda x: x[1])
	df['std_blue'] = df['std_rgb'].apply(lambda x: x[2])
	df = df[['fake_cnt', 'real_cnt', 
		     'avg_size', 'min_size', 'max_size', 
			 'mean_red', 'mean_green', 'mean_blue', 
			 'std_red', 'std_green', 'std_blue']].T
	st.table(df)

	st.write(response_eda)
