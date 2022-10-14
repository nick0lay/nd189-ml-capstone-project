import argparse
import json
import logging
import os
import sys
import io

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle

from data_processing import create_loaders
from model import TransformerTime2Vec
from training import train_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

model_file='model.pth'
arguments_file='args.p'
deafult_feature_columns=['Adj Close']

# Load model
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model
def model_fn(model_dir):
    logger.info(f"Creating model from directory {model_dir}")
    
    # load arguments
    path_args = os.path.join(model_dir, arguments_file)
    args = pickle.load(open(path_args, "rb"))
    logger.info(f"Loaded arguments: {args}")
    
    # creae model
    model = net(args)
    with open(os.path.join(model_dir, model_file), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def save_model(model, model_dir):
    logger.info(f"Saving the model in {model_dir}.")

# Takes the deserialized request object and performs inference against the loaded model.
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#get-predictions-from-a-pytorch-model
def predict_fn(input_data, model):
    logger.info("Predicting ...")

# Takes request data and deserializes the data into an object for prediction.
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#process-model-input
def input_fn(request_body, request_content_type, context):
    logger.info("Transform input data")

# Takes the result of prediction and serializes this according to the response content type.
def output_fn(prediction, content_type):
    logger.info("Transform model output data")

def net(args):
    return TransformerTime2Vec(feature_size=len(args.feature_columns), use_mask=True)

def main(args):
    logger.info(f'Start training with args: {args}')

    # load stock dataset
    data = pd.read_csv(args.data + '/stock.csv')
    # training sequence length
    seq_len = 7 # in days
    # validation dataset length in days
    valid_len = 105 # in days
    # test dataset length in days
    test_len = 105 # in days

    epochs=3
    lr=args.learning_rate

    # configure feature scalers
    target_column = 'Adj Close'
    feature_columns = args.feature_columns
    scalers = [
        (['Adj Close'], MinMaxScaler()),
    ]

    train_loader, valid_loader, test_loader = create_loaders(data[feature_columns], target_column, scalers, valid_len, test_len)
    model = net(args)
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, epochs, lr)

    path = os.path.join(args.model_dir, model_file)
    torch.save(model.state_dict(), path)
    # save arguments
    path_args = os.path.join(args.model_dir, arguments_file)
    pickle.dump(args, open(path_args, "wb"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--feature_columns', nargs='+', type=str)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_DATA'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()

    main(args)