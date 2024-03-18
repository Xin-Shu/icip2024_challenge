FROM tensorflow/tensorflow:2.13.0-gpu
ENV PATH=/usr/local/cuda/bin:$PATH
RUN apt-get update
RUN apt-get -y install libmediainfo-dev
WORKDIR /app
COPY . .
RUN pip install -r ./requirements.txt
ENTRYPOINT ["python3", "byte_predictor.py"]