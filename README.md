# Reduction-Synthesis: Plug-and-Play for Sentiment Style Transfer [:link:](https://aclanthology.org/2024.inlg-main.28/)

Yelp, Amzon [datasets](https://github.com/suzgunmirac/prompt-and-rerank/tree/main/datasets) are used in our experiments.
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



## Citation
```
@inproceedings{xu-etal-2024-reduction-synthesis,
    title = "Reduction-Synthesis: Plug-and-Play for Sentiment Style Transfer",
    author = "Xu, Sheng  and
      Fukumoto, Fumiyo  and
      Suzuki, Yoshimi",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.28",
    pages = "330--343",
}

```