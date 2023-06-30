import pyworkflow.protocol.params as params
import multiprocessing as mp
from functools import partial

from cryomethods import Plugin
from cryomethods.functions import NumpyImgHandler
from .protocol_base import ProtocolBase
from cryomethods.functions import num_flat_features, calcAvgPsd
from pwem.objects import CTFModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json


class Protdctf(ProtocolBase):
    """
    Calculate the CTF with deep learning.
    """
    _label = 'dctf'

    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)

    # --------------- DEFINE param functions ---------------

    def _defineParams(self, form):
        form.addSection('Params')

        group = form.addGroup('General')
        group.addParam('predictEnable', params.BooleanParam, default=False,
                       label='Predict',
                       help='Set true if you want to predict, set false if you want to train.')
        group.addParam('useGPU', params.BooleanParam, default=True,
                       label='Use GPU',
                       help='Enable the use of the GPU')
        group.addParam('batch_size', params.IntParam, default=100,
                       label='Batch size',
                       help='Number of images per iteration.')
        group = form.addGroup('Predict', condition="predictEnable")
        group.addParam('inputImgs', params.PointerParam, allowsNull=True,
                       pointerClass='SetOfMicrographs',
                       label="Input micrographs",
                       help='Select the input images.')
        group.addParam('weightsfile', params.PathParam,
                       label='Weigths file:',
                       help='Select the weights file of the neuronal network model.')

        group = form.addGroup('Train', condition="not predictEnable")
        group.addParam('trainSet', params.PointerParam, allowsNull=True,
                       pointerClass='SetOfCTF',
                       label="Train set",
                       help='Select the train images.')
        group.addParam('transferLearning', params.BooleanParam, default=True,
                       label='Transfer Learning',
                       help='Enable if you want to train using a pretrained model.')
        group.addParam('pretrainedModel', params.PathParam, condition="transferLearning",
                       label='Pretrained Model:',
                       help='Select the weights file of the pretrained neuronal network model.')
        group.addParam('lr', params.FloatParam, default=0.0001,
                       label='Learning rate',
                       help='Learning rate.')
        group.addParam('validation_ratio', params.FloatParam, default=0.20, expertLevel=params.LEVEL_ADVANCED,
                       label='Validation ratio',
                       help='Percentage of the validation set.')
        group.addParam('validation_split', params.BooleanParam, default=True, expertLevel=params.LEVEL_ADVANCED,
                       label='Validation splitting',
                       help='If True the splitting is done randomply if False the last images are assigned to validation.')

        group.addParam('epochs', params.IntParam, default=100,
                       label='Number of epochs',
                       help='The number of epochs for the training.')
        group.addParam('weightEveryEpoch', params.BooleanParam, default=True,
                       label='Save weight per epoch',
                       help='Enable if you want to save the weights per epoch, otherwise it will only save the weight of the las epoch.')

        form.addParallelSection(threads=1, mpi=1)
        form.addSection('PSD')
        group = form.addGroup('PSD estimation')
        group.addParam('window_size', params.IntParam, default=256,
                       label='Window size:',
                       help='Window size of the estimated PSD.')
        group.addParam('step_size', params.IntParam, default=128,
                       label='Step size:',
                       help='Size of the step in the periodogram averaging to estimate the PSD.')
        group.addParam('sampling', params.FloatParam, default=2.0,
                       label='Set sampling rate to:',
                       help='All images will be transformed to this sampling rate to train and predict.')

    # --------------- INSERT steps functions ----------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('runCTFStep')
        self._insertFunctionStep('createOutputStep')

    # --------------- STEPS functions -----------------------

    def convertInputStep(self):
        if self.predictEnable:
            self.imgSet = self.inputImgs.get()
            self.sampling_rate = self.imgSet.getSamplingRate()
            self.images_path = []
            for i, img in enumerate(self.imgSet):
                loc = img.getLocation()
                self.images_path.append(loc[1])

        else:
            self.ctfs = self.trainSet.get()
            self.data = []
            self.images_path = []
            path_psd = self._getExtraPath()
            sampling = self.sampling.get()

            ctfs  = []
            for i, ctf in enumerate(self.ctfs):
                extended_ctf = CTFModel()
                extended_ctf.set(ctf)
                dic_ctf = extended_ctf.getObjDict()
                filename_img = dic_ctf['_objValue._micObj._filename']
                sampling_rate = dic_ctf['_objValue._micObj._samplingRate']
                defocus = np.asarray(ctf.getDefocus())
                resolution = np.asarray(ctf.getResolution())
                ctfs.append([defocus, resolution, filename_img, sampling_rate])

            nthreads = max(1, self.numberOfThreads.get() * self.numberOfMpi.get())
            pool = mp.Pool(processes=nthreads)

            args = partial(process_ctf, path_psd=path_psd, sampling=sampling, window_size=self.window_size.get(),
                           step_size=self.step_size.get())
            results = pool.map(args, ctfs)

            self.data.extend(results)
            pool.close()
            pool.join()

    def runCTFStep(self):

        if self.predictEnable:
            self.psd_list, self.results = self.predict_CTF(self.images_path, self.window_size.get())
        else:
            self.train_nn(self.data)

    def createOutputStep(self):
        if self.predictEnable:
            self.ctfResults = self._createSetOfCTF()
            for i, img in enumerate(self.imgSet):
                ctf = CTFModel()
                ctf.setResolution(self.results[i][3])
                ctf.setMicrograph(img)
                ctf.setPsdFile(self.psd_list[i])
                ctf.setStandardDefocus(self.results[i][0], self.results[i][1], self.results[i][2])
                self.ctfResults.append(ctf)
            self._defineOutputs(ctfResults=self.ctfResults)

    # --------------- INFO functions -------------------------

    def _validate(self):
        return []

    def _citations(self):
        return []

    def _summary(self):
        return []

    def _methods(self):
        return []

    # --------------- UTILS functions ------------------------

    def calc_psd_per_mic(self, filename_img):
        img = NumpyImgHandler.loadMrc(filename_img)
        img = img[0, :, :]

        # Resize image if necesary to adjust downsampling
        new_size = (int(img.shape[1] * self.sampling_rate / self.sampling.get()),
                    int(img.shape[0] * self.sampling_rate / self.sampling.get()))
        PIL_image = Image.fromarray(img)
        resized_image = PIL_image.resize(new_size, resample=Image.BICUBIC)
        img = np.asarray(resized_image)

        # Calculate psd:
        psd = calcAvgPsd(img, windows_size=self.window_size.get(), step_size=self.step_size.get())
        filename_psd = self._getExtraPath() + '/' + os.path.basename(filename_img) + '_psd.mrc'
        NumpyImgHandler.saveMrc(psd, filename_psd)
        return filename_psd

    def train_nn(self, data):
        """
        Method to create the model and train it
        """
        # Especifica el porcentaje del conjunto de validación
        validation_ratio = self.validation_ratio.get()
        # Calcula el tamaño del conjunto de validación
        validation_size = int(validation_ratio * len(data))
        # Calcula el tamaño del conjunto de entrenamiento
        train_size = len(data) - validation_size

        # Divide el dataset en conjuntos de entrenamiento y validación

        if (self.validation_split.get):
            train_dataset, val_dataset = random_split(data, [train_size, validation_size])
        else:
            # Obtener los índices del conjunto de datos
            indices = list(range(len(data)))

            # Dividir los índices en subconjuntos de acuerdo con las proporciones
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Utilizar los índices para obtener los subconjuntos correspondientes
            train_dataset = Subset(data, train_indices)
            val_dataset = Subset(data, val_indices)

        # Crea los DataLoaders para entrenamiento y validación
        nthreads = max(1, self.numberOfThreads.get() * self.numberOfMpi.get())
        trainset = LoaderTrain(train_dataset, self._getExtraPath(), self.window_size.get(), self.step_size.get())
        data_loader_training = DataLoader(trainset, batch_size=self.batch_size.get(), shuffle=True,
                                          num_workers=nthreads, pin_memory=True)

        valset = LoaderTrain(val_dataset, self._getExtraPath(), self.window_size.get(), self.step_size.get())
        data_loader_val = DataLoader(valset, batch_size=self.batch_size.get(), shuffle=False, num_workers=nthreads,
                                     pin_memory=False)

        print('Total data training... {}'.format(len(data_loader_training.dataset)))
        print('Total data validation... {}'.format(len(data_loader_val.dataset)))

        # Set device
        use_cuda = self.useGPU and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', device)

        # Create the model
        model = Regresion(size_in=(1, self.window_size.get(), self.window_size.get()), size_out=4)

        if self.transferLearning.get():
            model.load_state_dict(torch.load(self.pretrainedModel.get()))

        model = model.to(device)
        print('Model:', model)

        optimizer = optim.Adam(model.parameters(), lr=self.lr.get())

        loss_function = weighted_mse_loss
        # criterion_test = nn.MSELoss(reduction = 'sum')

        self.loss_list_training = []
        self.loss_list_val = []

        for epoch in range(1, self.epochs.get() + 1):
            print('\nEpoch:', epoch, '/', self.epochs.get())
            train(model, device, data_loader_training, optimizer, loss_function)

            if self.weightEveryEpoch:
                torch.save(model.state_dict(),
                           os.path.join(self._getExtraPath(), 'model_weights' + str(epoch) + '.pt'))  # JV
                model = model.to(device)

            loss = self.calcLoss(model, data_loader_training, device, loss_function)
            self.loss_list_training.append(loss)
            print('Loss epoch training: {:.6f}'.format(loss))

            if (epoch % 10 == 0):
                loss_val = self.calcLoss(model, data_loader_val, device, loss_function)
                self.loss_list_val.append(loss_val)
                print('Loss epoch validation: {:.6f}'.format(loss_val))
                self.plot_loss_screening(self.loss_list_training, self.loss_list_val)
            else:
                self.loss_list_val.append(np.nan)

        if not self.weightEveryEpoch:
            model.train()
            torch.save(model.state_dict(), os.path.join(self._getExtraPath(), 'model_weights.pt'))  # JV

        print("Training loss")
        print(self.loss_list_training)

        print("Validation loss")
        print(self.loss_list_val)

        self.plot_loss_screening(self.loss_list_training, self.loss_list_val)

    def plot_loss_screening(self, loss_list_traning, loss_list_val):
        plt.figure(figsize=(11, 8))
        plt.plot(loss_list_traning, marker='o', linestyle='-', color='blue', label='Loss training')
        plt.plot(loss_list_val, marker='o', color='red', label='Loss validation')
        plt.title('Loss function')
        plt.ylabel('Loss function')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._getPath() + '/' + 'loss.png')

    def predict_CTF(self, images_path, window_size):
        """
        Method to prepare the model and calculate the CTF of the psd
        """
        trainset = LoaderPredict(images_path, self.weightsfile.get(), self.window_size.get(), self.step_size.get(),
                                 self.sampling_rate, self.sampling.get())

        nthreads = max(1, self.numberOfThreads.get() * self.numberOfMpi.get())
        data_loader = DataLoader(trainset, batch_size=self.batch_size.get(), shuffle=False, num_workers=nthreads,
                                 pin_memory=False)

        print('Total data... {}'.format(len(data_loader.dataset)))

        # Set device
        use_cuda = self.useGPU and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', device)

        # Create the model and load weights
        model = Regresion(size_in=(1, window_size, window_size), size_out=4)

        # Crear una instancia del modelo
        # model = Net()

        # Configurar el modelo en modo de evaluación
        # model.eval()

        # Realizar múltiples pasadas hacia adelante con Dropout activado
        # num_samples = 100
        # predictions = []
        # for _ in range(num_samples):
        #    output = model(input)
        #    predictions.append(output)

        # Calcular la predicción final promediando las salidas
        # prediction_mean = torch.mean(torch.stack(predictions), dim=0)

        model.load_state_dict(torch.load(self.weightsfile.get()))

        model = model.to(device)
        return predict(model, device, data_loader, trainset, self.batch_size.get(), self._getExtraPath())

    def calcLoss(self, model, data_loader, device, loss_function):
        """
        Calculate the value of the loss function
        """
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                # Move tensors to the configured device
                data, target = data['image'].to(device), data['target'].to(device)
                # Forward pass
                output = model(data)

                # Sum up batch loss
                test_loss += loss_function(output, target).item()

        num_batches = len(data_loader.dataset) / data_loader.batch_size
        return test_loss / num_batches


def predict(model, device, data_loader, trainset, batch_size, extraPath):
    """
    Method to predict using the neuronal network
    """
    model.eval()
    results = []
    psd_list = []

    with torch.no_grad():
        for data in data_loader:
            # batch size in size
            size = data['image'].shape[0]
            for batch_id in range(0, size):
                filename = extraPath + '/' + os.path.basename(data['name'][batch_id]) + '_psd.mrc'
                image = data['image']
                NumpyImgHandler.saveMrc(np.float32(data['image'][batch_id, :, :, :]), filename)

            # Move tensors to the configured device
            image = image.to(device)
            # Forward pass
            output = model(image)
            output = trainset.normalization.inv_transform(output.cpu().numpy())
            # Save results

            for batch_id in range(0, size):
                results.append(output[batch_id])
                filename = extraPath + '/' + os.path.basename(data['name'][batch_id]) + '_psd.mrc'
                psd_list.append(filename)

    return psd_list, results


def train(model, device, train_loader, optimizer, loss_function):
    """
    Method to train the neuronal network
    """
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # Move tensors to the configured device
        data, target = data['image'].to(device), data['target'].to(device)

        # Forward pass
        output = model(data)
        loss = loss_function(output, target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print data
        if batch_idx % 5 == 0:
            print('Train: [{}/{} ({:.0f}%)]    \tLoss: {:.6f}'.format(
                batch_idx * train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


'''
import torch
import torch.nn as nn
import torchvision.models as models

# Definir una subclase personalizada de ResNet-50
class ResNetRegression(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetRegression, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Eliminar la última capa lineal

        # Agregar una capa lineal personalizada para la regresión
        self.regression_layer = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression_layer(x)
        return x

# Crear una instancia del modelo personalizado
model = ResNetRegression()

# Imprimir la estructura del modelo
print(model)
'''


class Regresion(nn.Module):
    """
    Neuronal Network model
    """
    def __init__(self, size_in=(1, 256, 256), size_out=4):
        super(Regresion, self).__init__()

        self.Conv2d_1a_3x3 = nn.Conv2d(size_in[0], 32, kernel_size=3, stride=2)
        self.bn_1a_3x3 = nn.BatchNorm2d(32)

        self.Conv2d_2a_3x3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn_2a_3x3 = nn.BatchNorm2d(32)

        self.Conv2d_2b_3x3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_2b_3x3 = nn.BatchNorm2d(64)

        self.Conv2d_3b_1x1 = nn.Conv2d(64, 80, kernel_size=1)
        self.bn_3b_1x1 = nn.BatchNorm2d(80)

        self.Conv2d_4a_3x3 = nn.Conv2d(80, 192, kernel_size=3)
        self.bn_4a_3x3 = nn.BatchNorm2d(192)

        self.flat_size = num_flat_features(self._get_conv_ouput(size_in))

        self.fc1 = nn.Linear(self.flat_size, 400)
        self.fc2 = nn.Linear(400, size_out)

    def _get_conv_ouput(self, shape):
        f = torch.rand(1, *shape)
        g = self._forward_conv(f)
        return g

    def _forward_conv(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.bn_1a_3x3(x)
        x = F.dropout2d(x)
        x = F.gelu(x)

        x = self.Conv2d_2a_3x3(x)
        x = self.bn_2a_3x3(x)
        x = F.dropout2d(x)
        x = F.gelu(x)

        x = self.Conv2d_2b_3x3(x)
        x = self.bn_2b_3x3(x)
        x = F.dropout2d(x)
        x = F.gelu(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Conv2d_3b_1x1(x)
        x = self.bn_3b_1x1(x)
        x = F.dropout2d(x)
        x = F.gelu(x)

        x = self.Conv2d_4a_3x3(x)
        x = self.bn_4a_3x3(x)
        x = F.dropout2d(x)
        x = F.gelu(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.flat_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LoaderPredict(Dataset):
    """
    Class to load the dataset for predict
    """
    def __init__(self, datafiles, weight_path, window_size, step_size, sampling_rate, sampling):
        super(LoaderPredict, self).__init__()
        Plugin.setEnviron()

        normalization_path = os.path.dirname(weight_path) + '/training_normalization.json'
        self.normalization = Normalization(None, None)
        self.normalization.load(normalization_path)

        self._data = [i for i in datafiles]
        self._window_size = window_size
        self._step_size = step_size
        self._sampling_rate = sampling_rate
        self._sampling = sampling

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img_path = self._data[index]
        img = self.open_image(img_path)
        img.unsqueeze_(0)
        return {'image': img, 'name': img_path}

    def open_image(self, filename):
        img = NumpyImgHandler.loadMrc(filename)
        img = img[0, :, :]

        # Resize image if necesary to adjust downsampling
        new_size = (
        int(img.shape[1] * self._sampling_rate / self._sampling), int(img.shape[0] * self._sampling_rate / self._sampling))
        PIL_image = Image.fromarray(img)
        resized_image = PIL_image.resize(new_size, resample=Image.BICUBIC)
        img = np.asarray(resized_image)

        psd = calcAvgPsd(img, windows_size=self._window_size, step_size=self._step_size)
        return torch.from_numpy(np.float32(psd))


class LoaderTrain(Dataset):
    """
    Class to load the dataset for train
    """
    def __init__(self, data, extra_path, window_size, step_size):
        super(LoaderTrain, self).__init__()
        Plugin.setEnviron()

        self.normalization = Normalization(data, extra_path)

        dataMatrix = np.array([d['target'] for d in data])
        dataMatrix = self.normalization.transform(dataMatrix)

        for i in range(len(data)):
            data[i]['target'] = dataMatrix[i]

        self._data = data
        self._window_size = window_size
        self._step_size = step_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img_path = self._data[index]['img']
        target = self._data[index]['target']
        img = self.open_image(img_path)
        return {'image': img, 'target': target, 'name': img_path}

    def open_image(self, filename):
        img = NumpyImgHandler.load(filename)
        return torch.from_numpy(img)


class Normalization:

    def __init__(self, data, extra_path):

        if data is None:
            return

        dataMatrix = self.process_data(data)

        self.set_max_min(dataMatrix)
        dataMatrix = self.scale(dataMatrix)
        self.set_mean_std(dataMatrix)
        filename = extra_path + '/training_normalization.json'

        if os.path.exists(filename):
            pass
        else:
            self.save(filename=extra_path + '/training_normalization.json')

    def process_data(self, data):
        dataMatrix = []
        for row in data:
            dataMatrix.append(row['target'])
        return np.array(dataMatrix)

    def set_max_min(self, dataMatrix):
        self._min_value = dataMatrix.min(axis=0)
        self._max_value = dataMatrix.max(axis=0)

        print(self._min_value)

    def set_mean_std(self, dataMatrix):
        self._mean = dataMatrix.mean(axis=0)
        self._std = dataMatrix.std(axis=0)

    def scale(self, data):
        return (data - self._min_value) / (self._max_value - self._min_value)

    def standard_score(self, data):
        return (data - self._mean) / self._std

    def inv_scale(self, data):
        return data * (self._max_value - self._min_value) + self._min_value

    def inv_standard_score(self, data):
        return data * self._std + self._mean

    def transform(self, data):
        data = self.scale(data)
        return self.standard_score(data)

    def inv_transform(self, data):
        data = self.inv_standard_score(data)
        return self.inv_scale(data)

    def save(self, filename):
        data = {'min': self._min_value.tolist(), 'max': self._max_value.tolist(
        ), 'mean': self._mean.tolist(), 'std': self._std.tolist()}
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load(self, filename):
        with open(filename) as file:
            data = json.load(file)
            self._min_value = np.array(data['min'], dtype=np.float32)
            self._max_value = np.array(data['max'], dtype=np.float32)
            self._mean = np.array(data['mean'], dtype=np.float32)
            self._std = np.array(data['std'], dtype=np.float32)

    def load_hardcoded(self):
        self._min_value = -1
        self._max_value = 1
        self._mean = 0.5
        self._std = 1

    def print_values(self):
        print('Min:', self._min_value)
        print('Max:', self._max_value)
        print('Mean:', self._mean)
        print('Std:', self._std)


def weighted_mse_loss(input, target):
    # weight = 10 * torch.abs(target[:, 0] - target[:, 1])
    # weight = 10 * torch.square(target[:, 0] - target[:, 1])
    weight = 1 - torch.exp(
        -1000 * (torch.abs(target[:, 0] - target[:, 1]) / torch.max(target[:, 0], target[:, 1])) ** 2)
    loss = (input - target) ** 2
    loss[:, 2] = weight * loss[:, 2]
    return torch.sum(loss) / len(loss)


def process_ctf(ctf, path_psd, sampling=2, window_size=256, step_size=128):
    defocus, resolution, filename_img, sampling_rate = ctf
    defocus = np.asarray(defocus)
    resolution = np.asarray(resolution)
    target = list(defocus)
    target.append(resolution)

    print(filename_img)
    filename_psd = calc_psd_per_mic_fast(filename_img, path_psd, sampling_rate, sampling, window_size, step_size)
    print(filename_psd)
    print("--------------------------------------------------------")

    return {'img': filename_psd, 'target': np.array(target, dtype=np.float32)}


def calc_psd_per_mic_fast(filename_img, path_psd, sampling_rate, sampling, window_size, step_size):
    img = NumpyImgHandler.loadMrc(filename_img)
    img = img[0, :, :]
    new_size = (int(img.shape[1] * sampling_rate / sampling), int(img.shape[0] * sampling_rate / sampling))
    PIL_image = Image.fromarray(img)
    resized_image = PIL_image.resize(new_size, resample=Image.BICUBIC)
    img = np.asarray(resized_image)

    # Calculate psd with some extra noise for data augmentation
    psd = calcAvgPsd(img, window_size, step_size, add_noise=True)
    filename_psd = path_psd + '/' + os.path.splitext(os.path.basename(filename_img))[0] + '_psd.mrc'
    NumpyImgHandler.saveMrc(psd, filename_psd)

    return filename_psd
