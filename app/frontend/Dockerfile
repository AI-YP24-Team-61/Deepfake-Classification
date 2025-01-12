FROM python:3.10
WORKDIR /app/frontend
COPY requirements.txt app/frontend/requirements.txt
COPY . /app/frontend
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
#CMD ["streamlit", "run", "streamlit_app.py"]
ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]