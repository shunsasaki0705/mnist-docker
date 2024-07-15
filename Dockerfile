# base image
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# copy requirements.txt from host to container
COPY requirements.txt requirements.txt

# install library
RUN pip install --no-cache-dir -r requirements.txt

# copy app file to the container
COPY . .

# execute app
CMD ["python", "mnist.py"]