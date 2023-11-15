# define base image to use
FROM python:3.10

# define directory
WORKDIR /app

# copy fiels to directry
COPY . /app

# install python dependencies
RUN pip install -r /app/requirements.txt

# run python script
CMD ["python", "main.py", "--learning_rate", "0.00001", "--warmup_steps", "100", "--weight_decay", "0.001"]