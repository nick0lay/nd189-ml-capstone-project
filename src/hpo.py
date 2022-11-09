import argparse
import json
import logging
import os
import sys
import io

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle

from data_processing import create_loaders
from model import TransformerTime2Vec
from training import train_model

from smdebug.pytorch import get_hook
# import smdebug as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

model_file='model.pth'
arguments_file='args.pkl'
scaler_file='scaler.pkl'

predict_scaler = None

deafult_feature_columns=['Adj Close']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model
def model_fn(model_dir):
    logger.info(f"Creating model from directory {model_dir}")
    
    # load arguments
    path_args = os.path.join(model_dir, arguments_file)
    args = pickle.load(open(path_args, "rb"))
    logger.info(f"Loaded arguments: {args}")

    # load and set scaler, will be used for later data tranformation
    path_scaler = os.path.join(model_dir, scaler_file)
    global predict_scaler
    predict_scaler = pickle.load(open(path_scaler, "rb"))
    logger.info(f"Scaler set and loaded.")
    
    # creae model
    model = net(args)
    with open(os.path.join(model_dir, model_file), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def save_model(model, model_dir):
    logger.info(f"Saving the model in {model_dir}.")

# Takes the deserialized request object and performs inference against the loaded model.
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#get-predictions-from-a-pytorch-model
def predict_fn(input_data, model, context):
    logger.info("Predicting ...")
    logger.info(type(input_data))
    logger.info(input_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seq_len = 7
    use_mask = True

    # generate mask
    mask = None
    if use_mask:
        mask = generate_square_subsequent_mask(seq_len).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device), mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Takes request data and deserializes the data into an object for prediction.
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#process-model-input
def input_fn(request_body, request_content_type, context):
    logger.info("Transform input data ...")
    logger.info(f"Request: {request_body}, content type: {request_content_type}")
    data = json.loads(request_body)
    # data = np.load(io.BytesIO(request_body))
    # data = data.tolist()
    # return torch.Tensor(list(data))
    scaled_data = predict_scaler.transform(data[0])
    scaled_data = np.expand_dims(scaled_data, axis=0)
    logger.info(f"Transformed data: {scaled_data}")
    print(type(data))
    print(type(scaled_data))
    # return torch.from_numpy(np.array(data))
    return torch.from_numpy(scaled_data).float()

# Takes the result of prediction and serializes this according to the response content type.
def output_fn(prediction, content_type, context):
    logger.info("Transform model output data ...")
    logger.info(f"Data: {prediction} with content type: {content_type}")
    value = prediction.cpu().numpy().astype(float)
    scaled_value = predict_scaler.inverse_transform(value)
    return json.dumps(scaled_value.tolist())

def net(args):
    model = TransformerTime2Vec(feature_size=len(args.feature_columns), use_mask=True)
    model = model.to(device)
    return model

def create_feature_scalers(feature_columns):
    scalers = [
        (['Adj Close'], MinMaxScaler()),
    ]
    return scalers

def main(args):
    hook = get_hook(create_if_not_exists=True)
    logger.info(f"Hook created {hook}")
    logger.info(f'Start training with args: {args}')
    # hook = smd.Hook(out_dir=args.output_dir)
    # logger.info(f"Hook created {hook}")

    # load stock dataset
    data = pd.read_csv(args.data + '/stock.csv')
    # training sequence length
    seq_len = 7 # in days
    # validation dataset length in days
    valid_len = 105 # in days
    # test dataset length in days
    test_len = 105 # in days

    epochs=args.epochs
    lr=args.learning_rate

    # configure feature scalers
    target_column = 'Adj Close'
    feature_columns = args.feature_columns
    scaler = MinMaxScaler()
    scalers = [
        (['Adj Close'], scaler),
    ]

    train_loader, valid_loader, test_loader = create_loaders(data[feature_columns], target_column, scalers, valid_len, test_len)
    model = net(args)
    if hook:
        hook.register_hook(model)
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, epochs, lr, use_mask=args.use_mask, device=device)

    path = os.path.join(args.model_dir, model_file)
    torch.save(model.to(device).state_dict(), path)
    
    # save arguments
    path_args = os.path.join(args.model_dir, arguments_file)
    pickle.dump(args, open(path_args, "wb"))
    
    # save scaler
    path_scaler = os.path.join(args.model_dir, scaler_file)
    pickle.dump(scaler, open(path_scaler, "wb"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--feature_columns', nargs='+', type=str)
    parser.add_argument('--use_mask', type=bool, default=True)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_DATA'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()

    logger.debug(f"Arguments: {args}")

    # fix issue with "feature_columns" args provided by sagemaker
    features = args.feature_columns
    if len(features) == 1 and features[0].startswith('['):
        # parse values to list
        logger.info(f"Extract values from {features[0]}")
        args.feature_columns = json.loads(features[0])

    main(args)