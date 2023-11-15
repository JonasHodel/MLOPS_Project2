# MLOPS Project 2

Repository which includes all files for the second project from jonas hodel for the MLops course.

Clone the repository and follow the instructions below to get the project up and running.

## Install docker

- For windows: [Windows installation dokumentation](https://docs.docker.com/desktop/install/windows-install)
- For Linux: [Linux installation dokumentation](https://docs.docker.com/desktop/install/linux-install)

## Create the docker image

create docker image:

```console
docker build -t jhodel/mlops .
```

## Run the docker image

run docker image:

```console
docker run -v C:\Repos\MLOPS_Project2\tb_logs:/app/tb_logs -itd jhodel/mlops
```

## Read trainings result

The tensorboard log files will be stored in the `tb_logs` folder. Start the tensorboard with the following statements:

```python
tensorboard --logdir=tb_logs 
```
