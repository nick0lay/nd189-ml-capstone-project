from hpo import model_fn
from hpo import input_fn
from hpo import output_fn
from hpo import predict_fn

def main():
    content_type = "application/json"
    model = model_fn('model')
    input = input_fn(str.encode("[[[10], [24], [56], [55], [23], [45], [77]]]"), content_type, {})
    # input = input_fn(str.encode("[[0.10, 0.24, 0.56, 0.55, 0.23, 0.45, 0.77]]"), content_type, {})
    prediction = predict_fn(input, model)
    output = output_fn(prediction, content_type)
    print(f"Output: {output}")

if __name__=='__main__':
    main()