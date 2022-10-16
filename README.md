# nd189-ml-capstone-project
Udacity "AWS Machine Learning Engineer Nanodegree Program" Capstone Project

## Local development
### Activate env
Activate environment
```
conda activate udacity
```
Deactivate environment
```
conda deactivate
```
### Test model training script
Set required env variables:
```
export SM_CHANNEL_DATA=data \
export SM_MODEL_DIR=model \
export SM_OUTPUT_DATA_DIR=output
```
Run training script
```
python src/hpo.py --learning_rate 0.00001 --data data --feature_columns "Adj Close" --epochs=2
```
