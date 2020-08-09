FROM continuumio/miniconda3:latest

RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install --upgrade pip
RUN pip install mlflow==1.10.0 boto3 awscli flask
# TODO: This is a hack to install sklearn. Properway is to make conda work and use conda to install sklearn
RUN pip install sklearn

RUN mkdir /app
RUN cd /app
RUN mkdir mlflow

COPY bin/run.sh /app/mlflow/run.sh
COPY app.py /app/mlflow/app.py

RUN chmod 777 /app/mlflow/run.sh

ENTRYPOINT ["/app/mlflow/run.sh"]
