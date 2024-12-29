import streamlit as st
import pandas as pd
from zipfile import ZipFile 

import plotly.express as px
import asyncio
from request_utils import post_data, get_data

import time

st.title("Team 61. Deepfake-Classification.")
st.header("Загрузка датасета", divider="gray")
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
    st.header(f"Профиль данных из {uploaded_file.name}", divider="gray")
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
    #st.write(response_eda)

st.header(f"Настройка нейронной сети", divider="gray")

id = st.text_input("Введите идентификатор (ID) для нейронной сети")
id = id.lower().replace(" ", "")

st.write(id)

type_nn_pretrain = 'ResNet18'

end_activation_function = "Sigmoid"

batch_size = st.select_slider("Выберите размер батча (batch_size)", 
                               options=[2 ** j for j in range(5, 11)])
    
lr = st.slider("Выберите размер шага обучения (learning rate)",
                min_value=0.01,
                max_value=0.1,
                step=0.01)
weight_decay = st.slider("Выберите величину коэффициента регуляризации ($\lambda$)",
              min_value=0.,
              max_value=1.,
              step=0.01)


left, middle, right = st.columns(3)
if left.button("Обучить нейронную сеть", use_container_width=True):
    if id != '' and id is not None:
        since = time.time()
        response_fit = asyncio.run(post_data('fit', input_data={
            'id': id,
            'hyperparameters' : {
                'type_nn_pretrain': type_nn_pretrain,
                'end_activation_function': end_activation_function,
                'batch_size': batch_size,
                'lr': lr,
                'C': weight_decay,
            }
        }))
        time_elapsed = time.time() - since
        if response_fit['message'] == f"Model with id '{id}' already exist. Enter another id.":
            st.write('Модель с таким ID уже существует. Введите другой ID.')
        else:
            left.write(f"Нейронная сеть обучена. Время обучения: {time_elapsed:.2f} секунд.")
            # response_models = list(asyncio.run(get_data('models'))[0].keys())
            # list_fitted_id = list(response_models[0].keys())

    else:
        left.markdown("Введите ID.")

with st.sidebar:
    if st.button("Вывести информацию об обученных моделях", use_container_width=False):
        response_models = asyncio.run(get_data('models'))
        if response_models[0]:
            st.write(response_models[0])
        else:
            st.write('Нет обученных моделей')


st.header(f"Инференс обученной модели", divider="gray")

if asyncio.run(get_data('models'))[0]:
    set_model_id = st.selectbox(
        "Выберите обученную модель для инференса",
        tuple(asyncio.run(get_data('models'))[0].keys())
    )

    if st.button(f"Установить '{set_model_id}' в качестве модели для инференса.", use_container_width=True, type='secondary'):
        asyncio.run(post_data('set', input_data={'id': set_model_id}))
        st.write(f"'{set_model_id}' установлена в качестве модели для инференса.")

    uploaded_file = st.file_uploader("Выберите изображение для инференса", type=['jpg'])
    if uploaded_file is not None:
    # Сохраняем zip-архив от пользователя в папку
        with open(fr"..\data\inference_image\inference_user_image.{uploaded_file.name.split('.')[1]}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, width=200)

        if st.button(f"PREDICT", use_container_width=True, type="primary"):
            predict_response = asyncio.run(post_data('predict', input_data={'id': set_model_id}))
            st.write(f"Вероятность принадлежности изображения к классу REAL: **{predict_response['real_prob']:.2f}**")
            if predict_response['is_real']:
                st.write("Изображение является **РЕАЛЬНЫМ**")
            else:
                st.write("Изображение является **ФЕЙКОВЫМ**")
                
else:
    st.write("Нет обученных моделей для инференса.")




