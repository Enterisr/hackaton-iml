# iml-hack-25

Welcome to the NBA manager challenge in IML hackathon 2025! <br>
Follow these instructions to setup your environment, ideally on a virtual environment.


## Data setup 

1. **Install requirements:**
```
pip install -r requirements.txt
```
3. **Download data:** follow instructions in [data/README.md](data/README.md)
4. **Switch to the [src](src) folder:**
```
cd ./src
```
6. **Create a sample of the train data:**
```
head -n 1 ../data/train_test_splits/train.jsonl > ./train.sample.jsonl
```

## Static draft baseline

1. **Predict on sample:**
```
python static_draft.py --in=./train.sample.jsonl --model=random --out=./
```
3. **Evaluate the predictions:**
```
python evaluate_predictions.py --gold=./train.sample.jsonl --pred=./RandomStaticBaseline_1.jsonl
```

This last step should print the score of a random team on a future season (probably between 1000-3000).

## Dynamic draft baseline
1. **Predict on sample:**
```
python dynamic_draft.py --in=./train.sample.jsonl --models=random,random,random --out=./
```

This will put three random baselines in a snake draft.

2. **Evaluate the predictions**
Each of the three models has made a draft, you can evaluate the first one for example by running:
```
python evaluate_predictions.py --gold=./train.sample.jsonl --pred=./RandomDynamicBaseline_1.jsonl
```
This last step should print the score of a random team on a future season (probably between 1000-3000).

## You're good to go

If this works, you're set to go!
Next read [src/static_draft.py](src/static_draft.py) and [src/dynamic_draft.py](src/dynamic_draft.py) to understand how this was achieved, and hack away.




