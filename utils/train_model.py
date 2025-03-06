import os
import time
import torch
from tempfile import TemporaryDirectory

def initialize_training(model):
    """
    Инициализация и сохранение модели перед началом обучения.
    """
    since = time.time()
    dict_stat = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    return since, dict_stat

def save_best_model(model, best_acc, epoch_acc, best_model_params_path):
    """
    Сохранение лучшей модели.
    """
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), best_model_params_path)
    return best_acc

def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1 - y_true * y_pred, min=0))

def process_phase(model, phase, dataloaders, criterion, optimizer, isLogreg, isSVM):
    """
    Обработка одной фазы (тренировка или валидация) на одной эпохе.
    """
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.cuda().to(torch.float32)
        labels = labels.cuda().to(torch.float32)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)

            if isLogreg:
                outputs = outputs.squeeze()
                preds = (outputs > 0.6).int().reshape(-1)
            elif isSVM:
                outputs = outputs.squeeze()
                margin = outputs * labels
                preds = (margin > 0).int().reshape(-1)

            if isLogreg:
                loss = criterion(outputs, labels)
            elif isSVM:
                loss = hinge_loss(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects

def update_statistics(dict_stat, phase, epoch_loss, epoch_acc):
    """
    Обновление статистик после обработки фазы.
    """
    dict_stat[f'{phase}_loss'].append(epoch_loss)
    dict_stat[f'{phase}_acc'].append(epoch_acc.item())
    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def train_epoch(model, dataloaders, criterion, optimizer, scheduler, isLogreg, isSVM, dataset_sizes, dict_stat, best_acc, best_model_params_path):
    """
    Обучение модели на одной эпохе.
    """
    for phase in ['train', 'test']:
        running_loss, running_corrects = process_phase(model, phase, dataloaders, criterion, optimizer, isLogreg, isSVM)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        update_statistics(dict_stat, phase, epoch_loss, epoch_acc)
        if phase == 'train':
            scheduler.step()
        if phase == 'test':
            best_acc = save_best_model(model=model, 
                                       best_acc=best_acc,
                                       epoch_acc=epoch_acc,
                                       best_model_params_path=best_model_params_path)
    return best_acc

def train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders=None, num_epochs=10, isLogreg=False, isSVM=False):
    """
    Основная функция для обучения модели.
    """
    since, dict_stat = initialize_training(model)

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        best_acc_list = [-1]
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            best_acc = train_epoch(model=model, 
                                   dataloaders=dataloaders, 
                                   criterion=criterion, 
                                   optimizer=optimizer, 
                                   scheduler=scheduler, 
                                   isLogreg=isLogreg, 
                                   isSVM=isSVM, 
                                   dataset_sizes=dataset_sizes, 
                                   dict_stat=dict_stat, 
                                   best_acc=max(best_acc_list),
                                   best_model_params_path=best_model_params_path)
            best_acc_list.append(best_acc)
            
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

    return model, dict_stat