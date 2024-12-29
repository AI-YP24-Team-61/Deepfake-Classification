import streamlit as st
import pandas as pd
from zipfile import ZipFile 

import plotly.express as px
import asyncio

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
