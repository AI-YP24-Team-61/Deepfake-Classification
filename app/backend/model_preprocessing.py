import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
from tqdm import tqdm
import numpy as np

# EPOCHS = 5
BATCH_SIZE = 128
PART_OF_DATASET = 8
EPOCHS = 1


class CustomModel(nn.Module):
    def __init__(self, lr=0.01, weight_decay=0.01):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        # self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        # Фиксируем все параметры нейронной сети. Будем настраивать параметры только в fc-слое
        for param in self.model.parameters():
            param.requires_grad = False
        # Для ResNet установить model.fc, для mobilnet_v3_small установить model.classifier || self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 1),
                                      nn.Sigmoid())
        self.model = self.model.cuda()
        # BCELoss - это обычный logloss. CrossEntropy для логистической регрессии использовать некорректно
        self.loss = torch.nn.BCELoss()
        # Для ResNet установить model.fc, для mobilnet_v3_small установить model.classifier
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def forward(self, x):
        return self.model(x)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train_model(model: CustomModel,
                id_model: str = 'test_model',
                dataloaders=None,
                dataset_sizes=None,
                num_epochs=10,
                ):
    """
    Функция для обучения нейронной сети.

    model - архитектура НС
    criterion - функция ошибки
    optimizer - вид градиентного спуска (SGD, Momentum, Adam)
    sheduler - объект, позволяющий итеративно изменять градиентный шаг
    num_epochs - число эпох
    """
    since = time.time()
    # Создаем словарь промежуточных статистик, чтобы считать качество на каждой эпохе
    dict_stat = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Create a temporary directory to save training checkpoints
    best_model_params_path = fr".\model_weights\{id_model}.pt"
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()   # Ставим eval-mode на тестовой выборке

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.cuda()
                inputs = inputs.to(torch.float32)

                labels = labels.cuda()
                labels = labels.to(torch.float32)

                model.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # Если вероятность принадлежности классу 1 больше 0.6, то относим объект к классу 1 (REAL)
                    if True:
                        outputs = outputs.squeeze()
                        # Можно поиграть с treshold. Конкретно этот показывает неплохой ACCURACY
                        preds = (outputs > 0.6).int().reshape(-1)

                    # Внутри любой функции Loss'а есть параметр reduction. Он отвечает за усреднение лосса по батчу.
                    if True:
                        loss = model.loss(outputs, labels)

                    # backward + optimize только на стадии обучения НС
                    if phase == 'train':
                        # Считаем вектор градиентов
                        loss.backward()
                        # Делаем градиентный шаг
                        model.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Заносим статистики в словарь
            dict_stat[f'{phase}_loss'].append(epoch_loss)
            dict_stat[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Сохраняем отдельно модель, которая дала лучшую accuracy
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
                print(best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model, dict_stat, best_acc


data_dir = r"..\data\cifake-real-and-ai-generated-synthetic-images"


def data_pipeline(data_dir=data_dir):
    image_datasets_logreg = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                     data_transforms[x]) for x in ['train', 'test']}
    evens = list(range(0, len(image_datasets_logreg['train']), PART_OF_DATASET))
    train_dataset = torch.utils.data.Subset(image_datasets_logreg['train'], evens)

    evens = list(range(0, len(image_datasets_logreg['test']), PART_OF_DATASET))
    test_dataset = torch.utils.data.Subset(image_datasets_logreg['test'], evens)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=True)

    dataset_sizes = {x: len(train_dataset) if x == 'train' else len(test_dataset) for x in ['train', 'test']}
    dataloaders_logreg = {x: train_loader if x == 'train' else test_loader for x in ['train', 'test']}
    return dataset_sizes, dataloaders_logreg


def predict_real(image_object, model):
    transform = transforms.Compose([
         transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im_tensor = transform(image_object)
    im_tensor = im_tensor.reshape(-1, 3, 32, 32)
    im_tensor = im_tensor.cuda()
    # В pred лежит вероятность отнесения к классу 1 (Real)
    pred = model(im_tensor)
    # Если True - значит Real.
    return {'real_prob': float(pred[-1][-1]),
            'is_real': float(pred[-1][-1]) > 0.6}


def calculate_statistics(data):
    dct_targetHist = {}
    for i_split in ['train', 'test']:
        # Тензоры для хранения прокси-информации по RGB каналам
        tsum = torch.tensor([0.0, 0.0, 0.0])
        tsum_sq = torch.tensor([0.0, 0.0, 0.0])
        # Переменные для расчета числа наблюдений в каждом классе
        cnt_fake = 0
        cnt_real = 0
        # Инициализируем списки для расчета статистик по размерам
        widths = []
        heights = []
        for image, target in tqdm(data[i_split]):
            # Прокси-расчеты для вычисления статистики по размерам изобрежений
            image_sz = torch.permute(image, (1, 2, 0))
            widths.append(image_sz.shape[0])  # Заносим в список параметры широты всех изображений
            heights.append(image_sz.shape[1])  # Заносим в список параметры широты всех изображений
            # Прокси-расчеты для вычисления статистики по mean/std
            tsum += image.sum(axis=(1, 2))
            tsum_sq += (image**2).sum(axis=(1, 2))
            # Считаем число экземпляров каждого класса в разрезе train/test
            if target == 1:
                cnt_real += 1
            elif target == 0:
                cnt_fake += 1
        if i_split == 'train':
            cntPixels = (np.array(widths) * np.array(heights)).sum()  # Количество пикселей в выборке
        elif i_split == 'test':
            cntPixels = (np.array(widths) * np.array(heights)).sum()  # Количество пикселей в выборке
        # Считаем статистики по размерам изображений
        array_sizes = np.multiply(widths, heights)  # Создаем массив всех размеров (width*heights)
        avg_size = np.mean(array_sizes)
        min_size = np.min(array_sizes)
        max_size = np.max(array_sizes)
        # Считаем mean/std в разрезе каждого класса
        mean_rgb = tsum / cntPixels
        var_rgb = (tsum_sq / cntPixels) - (mean_rgb**2)  # E(X^2) - E^2(X)
        std_rgb = torch.sqrt(var_rgb)
        dct_targetHist[i_split] = dct_targetHist.get(i_split,
                                                     {'fake_cnt': cnt_fake,
                                                      'real_cnt': cnt_real,
                                                      'avg_size': float(avg_size),
                                                      'min_size': float(min_size),
                                                      'max_size': float(max_size),
                                                      'mean_rgb': mean_rgb.tolist(),
                                                      'var_rgb': var_rgb.tolist(),
                                                      'std_rgb': std_rgb.tolist()})
    return dct_targetHist


def make_eda():
    # Предварительная загрузка и трансформация данных из локальной папки на ПК
    data = {
        # Создаем словарь, где по ключу train/test будут лежать соответствующие данные
        # По очереди обращаемся к папкам train/test, чтобы загрузить оттуда данные
        split: datasets.ImageFolder(
            # Если путь f"{DATASET_DIR}/{split}" содержит другие папки,
            # то названия этих папок устанавливаются в качестве
            # классов (LABELS) для данных в папке f"{DATASET_DIR}/{split}".
            # Узнать метки классов можно с помощью вызова data['train'].classes
            fr"..\data\dataset_example\{split}",
            #
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        for split in ['train', 'test']
    }
    print("Данные загружены!")
    return calculate_statistics(data)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device!")

    # Обучаем модель
    model = CustomModel(lr=0.01,
                        weight_decay=0.01)

    dataset_sizes, dataloaders_logreg = data_pipeline(data_dir=data_dir)
    model_inf, dict_stat = train_model(model,
                                       dataset_sizes=dataset_sizes,
                                       dataloaders=dataloaders_logreg,
                                       num_epochs=EPOCHS)
    print(dict_stat)
