## Requirement

ubuntu 22.04.1

docker 24.0.5

conda 24.4.0


## Setup

Stpe1: Execuate evaluator and ollama docker services

```
$ cd docker
$ docker compose up -d
```



Step2: Set local environment
```
$ conda create -n reduction-synthesis python=3.12
$ conda activate reduction-synthesis
$ pip install -r requirements.txt 
```
## Experiment
```
$ bash run.sh
```
The output is storaged at "output" dir

Note: the file docker/.env defines Environment Variables used in docker container and experiment code.