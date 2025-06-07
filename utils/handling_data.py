from matplotlib import pyplot as plt
import numpy as np



def summarize_diagnostics(history, text_title='Log Loss', title="ResNet18"):
    """
    Функция для отрисовки значений LOSS и ACCURACY на каждой эпохе
    """
    fig,ax=plt.subplots(2)
    #fig.suptitle("Loss")
    fig.tight_layout()
    # plot loss
    ax[0].set_title(f'{text_title}')
    ax[0].plot(history['train_loss'], color='blue', label='Training loss')
    ax[0].plot(history['test_loss'], color='orange', label='validation loss')
    ax[0].legend(loc='best', shadow=True)
    # plot accuracy
    ax[1].set_title('Classification Accuracy')
    ax[1].plot(history['train_acc'], color='blue', label='train')
    ax[1].plot(history['test_acc'], color='orange', label='test')
    ax[1].legend(loc='best', shadow=True)
    
def imshow(inp, title=None):
    """Отображаем батч картинок. Лучше всего печатать по 8 картинок. Так нагляднее."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
