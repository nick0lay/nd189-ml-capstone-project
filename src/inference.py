import json
import logging
import os
import sys

import numpy as np
import torch
import pickle

from model import TransformerTime2Vec

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
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
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