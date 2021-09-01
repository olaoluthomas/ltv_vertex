# Use the official lightweight Python image.
FROM python:3.9-slim
LABEL version = "v0.1"
LABEL maintainer="Simeon Thomas simeon.thomas@bedbath.com"
WORKDIR /root

# Copy files into docker image
COPY ./requirements.txt /root/requirements.txt
COPY ./src /root/src
COPY ./main.py /root/main.py

# Install production dependencies.
RUN pip install -r /root/requirements.txt

# Download training data.
RUN curl https://storage.cloud.google.com/dev_dw_npii_adhoc/analytics/LTV/data/btyd_training_data.csv --output /root/train_data.csv

# Set up the entry point to invoke the trainer.
ENTRYPOINT [ "python", "main.py" ]