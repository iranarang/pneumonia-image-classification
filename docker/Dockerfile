FROM python:3.9

WORKDIR /app

RUN pip install --no-cache-dir "numpy<2.0" tensorflow==2.17 Flask==3.0 Pillow==10.0

COPY docker/api.py /app/api.py  

CMD ["python", "api.py"]

