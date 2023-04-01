
import pyworkflow.protocol.params as params

from cryomethods import Plugin
from cryomethods.functions import NumpyImgHandler

from .protocol_base import ProtocolBase

from cryomethods.functions import num_flat_features, calcAvgPsd

from pwem.objects import CTFModel

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
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

        group = form.addGroup('Predict', condition="predictEnable")
        group.addParam('inputImgs', params.PointerParam, allowsNull=True,
                       pointerClass='SetOfMicrographs',
                       label="Input micrographs",
                       help='Select the input images.')
        group.addParam('weightsfile', params.PathParam,
                       label='Weigths file:',
                       help='Select the weights file of the neuronal network model.')

        group = form.addGroup('PSD estimation')
        group.addParam('window_size', params.IntParam, default=256,
                       label='Window size:',
                       help='Window size of the estimated PSD.')
        group.addParam('step_size', params.IntParam, default=128,
                       label='Step size:',
                       help='Size of the step in the periodogram averaging to estimate the PSD.')

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
        group.addParam('epochs', params.IntParam, default=20,
                       label='Number of epochs',
                       help='The number of epochs for the training.')
        group.addParam('weightEveryEpoch', params.BooleanParam, default=True,
                       label='Save weight per epoch',
                       help='Enable if you want to save the weights per epoch, otherwise it will only save the weight of the las epoch.')

    # --------------- INSERT steps functions ----------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('runCTFStep')
        self._insertFunctionStep('createOutputStep')

    # --------------- STEPS functions -----------------------

    def convertInputStep(self):
        if self.predictEnable:
            self.imgSet = self.inputImgs.get()
            self.images_path = self.imgSet.getFiles()
        else:
            self.ctfs = self.trainSet.get()
            self.data = []
            self.images_path = []

            for ctf in self.ctfs:

                target = list(ctf.getDefocus())
                target.append(ctf.getResolution())
                img = ctf.getPsdFile()
                self.data.append({'img':img, 'target': np.array(target, dtype=np.float32)})
                self.images_path.append(img)

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
            # self._defineSourceRelation(self.inputImgs, self.out)

    # --------------- INFO functions -------------------------

    def _validate(self):
        return []

    def _citations(self):
        return []

    def _summary(self):
        return []

    def _methods(self):
        return []

    # --------------- UTILS functions -------------------------

    def train_nn(self, data):
        """
        Method to create the model and train it
        """
        trainset = LoaderTrain(data,self.images_path, self.window_size.get(), self.step_size.get()) #JV
#       trainset = LoaderTrain(images_path,'/home/jvargas/ScipionUserData/projects/TestWorkflowRelion3Betagal/images.txt')  #/home/alex/cryoem/norm.txt')
        data_loader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=1, pin_memory=True)
        print('Total data... {}'.format(len(data_loader.dataset)))

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

        criterion_train = weighted_mse_loss
        criterion_test = nn.MSELoss(reduction = 'sum')

        self.loss_list = []
        self.accuracy_list = []

        for epoch in range(1, self.epochs.get() + 1):
            print('\nEpoch:', epoch, '/', self.epochs.get())
            train(model, device, data_loader, optimizer, criterion_train)
            if self.weightEveryEpoch:
                #torch.save(model.cpu().state_dict(), os.path.join(self.weightFolder.get(), 'model_' + str(epoch) + '.pt'))
                #torch.save(model.state_dict(), os.path.join(self.weightFolder.get(), 'model_' + str(epoch) + '.pt')) #JV
                torch.save(model.state_dict(), os.path.join(self._getExtraPath(), 'model_weights' + str(epoch) + '.pt')) #JV

                model = model.to(device)

            loss = self.calcLoss(model, data_loader, device, criterion_test)
            self.loss_list.append(loss)

        if not self.weightEveryEpoch:
            model.train()
            #torch.save(model.cpu(), os.path.join(self.weightFolder.get(), 'model.pt'))
            torch.save(model.state_dict(), os.path.join(self._getExtraPath(), 'model_weights.pt')) # JV

        print(self.loss_list)
        self.plot_loss_screening(self.loss_list)

    def plot_loss_screening(self, loss_list):
        """
        Create the plot figure using the values of loss_list
        """        
        plt.figure(figsize=(11, 8))
        plt.plot(loss_list)
        plt.title('Loss function')
        plt.ylabel('Loss function')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss.png')

    def predict_CTF(self, images_path, window_size):
        """
        Method to prepare the model and calculate the CTF of the psd
        """
        trainset = LoaderPredict(images_path, self.images_path, self.window_size.get(), self.step_size.get())
        #trainset = LoaderPredict(images_path, '/home/jvargas/ScipionUserData/projects/TestWorkflowRelion3Betagal/images.txt') #'/home/alex/cryoem/norm.txt')
        data_loader = DataLoader(trainset, batch_size=1,
                                 shuffle=False, num_workers=1, pin_memory=False)
        print('Total data... {}'.format(len(data_loader.dataset)))
        # Set device
        use_cuda = self.useGPU and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', device)
        # Create the model and load weights
        model = Regresion(size_in=(1, window_size, window_size), size_out=4)

        model.load_state_dict(torch.load(self.weightsfile.get()))

        model = model.to(device)
        return predict(model, device, data_loader, trainset, self._getExtraPath())

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

        return test_loss / len(data_loader.dataset)

def predict(model, device, data_loader, trainset, extraPath):
    """
    Method to predict using the neuronal network
    """
    model.eval()
    results = []
    psd_list = []
    with torch.no_grad():
        for data in data_loader:
            # Move tensors to the configured device
            #filename = 'psd/' + os.path.basename(data['name'][0]) + '_psd.mrc'
            filename = extraPath + '/' + os.path.basename(data['name'][0]) + '_psd.mrc'
            image = data['image']
            NumpyImgHandler.saveMrc(np.float32(image.numpy()), filename)
            image = image.to(device)
            # Forward pass
            output = model(image)
            output = trainset.normalization.inv_transform(output.cpu().numpy())
            # Save results
            results.append(output[0])
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


class Regresion(nn.Module):
    """
    Neuronal Network model
    """

    def __init__(self, size_in=(1, 419, 419), size_out=3):
        super(Regresion, self).__init__()

        self.Conv2d_1a_3x3 = nn.Conv2d(size_in[0], 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = nn.Conv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = nn.Conv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = nn.Conv2d(80, 192, kernel_size=3)

        self.flat_size = num_flat_features(self._get_conv_ouput(size_in))

        self.fc1 = nn.Linear(self.flat_size, 400)
        self.fc2 = nn.Linear(400, size_out)

    def _get_conv_ouput(self, shape):
        f = torch.rand(1, *shape)
        g = self._forward_conv(f)
        return g

    def _forward_conv(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = F.dropout2d(x)
        x = F.relu(x)

        x = self.Conv2d_2a_3x3(x)
        x = F.dropout2d(x)
        x = F.relu(x)

        x = self.Conv2d_2b_3x3(x)
        x = F.dropout2d(x)
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Conv2d_3b_1x1(x)
        x = F.dropout2d(x)
        x = F.relu(x)

        x = self.Conv2d_4a_3x3(x)
        x = F.dropout2d(x)
        x = F.relu(x)

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
    def __init__(self, datafiles, norm_file, window_size, step_size):
        super(LoaderPredict, self).__init__()
        Plugin.setEnviron()

        self.normalization = Normalization(None)
        #self.normalization.load(norm_file)
        self.normalization.load_hardcoded()

        self._data = [i for i in datafiles]
        self._window_size = window_size
        self.step_size = step_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img_path = self._data[index]
        img = self.open_image(img_path)
        img.unsqueeze_(0)
        return {'image': img, 'name': img_path}

    def open_image(self, filename):
        img = NumpyImgHandler.loadMrc(filename)
        # _min = img.min()
        # _max = img.max()
        # img = (img - _min) / (_max - _min)
        psd = calcAvgPsd(img[0,:,:], windows_size=self._window_size, step_size=self.step_size)
        # img = np.resize(img, (1, 512, 512))
        return torch.from_numpy(np.float32(psd))


class LoaderTrain(Dataset):
    """
    Class to load the dataset for train
    """
    def __init__(self, data, norm_file, window_size, step_size):
        super(LoaderTrain, self).__init__()
        Plugin.setEnviron()

        self.normalization = Normalization(None)
        #self.normalization.load(norm_file)

        dataMatrix = np.array([d['target'] for d in data])
        #dataMatrix = self.normalization.transform(dataMatrix)
        
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
        #img = NumpyImgHandler.loadMrc(filename)
        img = NumpyImgHandler.load(filename)
        _min = img.min()
        _max = img.max()
        img = (img - _min) / (_max - _min)
        img = np.resize(img, (1, self._window_size, self._window_size))
        return torch.from_numpy(img)


class Normalization():

    def __init__(self, dataMatrix):

        if dataMatrix is None:
            return

        self.set_max_min(dataMatrix)
        dataMatrix = self.scale(dataMatrix)
        self.set_mean_std(dataMatrix)

    def set_max_min(self, dataMatrix):
        self._min_value = dataMatrix.min(axis=0)
        self._max_value = dataMatrix.max(axis=0)

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
        self._std =  1

    def print_values(self):
        print('Min:', self._min_value)
        print('Max:', self._max_value)
        print('Mean:', self._mean)
        print('Std:', self._std)


def weighted_mse_loss(input, target):
    weight = 10 * torch.abs(target[:, 0] - target[:, 1])
    loss = (input - target) ** 2
    loss[:, 2] = weight * loss[:, 2]
    return torch.sum(loss)/len(loss)
