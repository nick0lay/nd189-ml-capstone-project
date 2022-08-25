import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

def plot_losses(train_loss, valid_loss):
    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Model loss (MSE)", fontsize=22)
    st.set_y(0.92)

    ax1 = fig.add_subplot(311)
    ax1.plot(train_loss[0:], label='Training loss (MSE)')
    ax1.plot(valid_loss[0:], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12) 

def draw_prediction(original, predictions=None):
    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Transformer predictions", fontsize=22)
    st.set_y(0.92)

    ax11 = fig.add_subplot(311)
    ax11.plot(original, label='Original')
    if not predictions is None:
        ax11.plot(predictions, linewidth=3, label='Predicted')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('Close price')
    ax11.legend(loc="best", fontsize=12)