import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.utils import (makePath, copyFile, replaceBaseExt)

from pyworkflow.object import Float

from cryomethods import Plugin
from cryomethods.convert import (writeSetOfParticles, rowToAlignment,
                                 relionToLocation, loadMrc, saveMrc,
                                 alignVolumes, applyTransforms)

from .protocol_base import ProtocolBase


import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

import os


class ProtSCNN(ProtocolBase):
    """
    Screening using a CNN.
    """
    _label = 'S-CNN'

    def __init__(self, **args):
        ProtocolBase.__init__(self, **args)

    # --------------- DEFINE param functions ---------------

    def _defineParams(self, form):

        form.addSection('Params')
        line = form.addLine('General')
        line.addParam('predictEnable', params.BooleanParam, default=False,
                      label='Predict',
                      help='Set true if you want to predict, set false if you want to train.')
        line.addParam('useGPU', params.BooleanParam, default=True,
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

        group = form.addGroup('Train', condition="not predictEnable")
        group.addParam('positiveInput', params.PointerParam, allowsNull=True,
                       pointerClass='SetOfMicrographs',
                       label="Positive micrographs",
                       help='Select the input images.')
        group.addParam('negativeInput', params.PointerParam, allowsNull=True,
                       pointerClass='SetOfMicrographs',
                       label="Negative micrographs",
                       help='Select the input images.')
        group.addParam('transferLearning', params.BooleanParam, default=True,
                       label='Transfer Learning',
                       help='Enable if you want to train using a pretrained model.')
        group.addParam('pretrainedModel', params.PathParam, condition="transferLearning",
                       label='Pretrained Model:',
                       help='Select the weights file of the pretrained neuronal network model.')
        group.addParam('weightFolder', params.PathParam,
                       label='Weight folder:',
                       help='Folder where the weights will be saved.')
        group.addParam('balance', params.BooleanParam, default=False,
                       label='Balance data',
                       help='Enable if you want to balance the dataset.')
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
        self._insertFunctionStep('runCNNStep')
        self._insertFunctionStep('createOutputStep')

    # --------------- STEPS functions -----------------------

    def convertInputStep(self):
        if self.predictEnable:
            self.imgSet = self.inputImgs.get()
            self.images_path = self.imgSet.getFiles()
        else:
            self.good_data = list(self.positiveInput.get().getFiles())
            self.bad_data = list(self.negativeInput.get().getFiles())

    def runCNNStep(self):
        if self.predictEnable:
            self.results = self.classsify_micrographs(self.images_path)
        else:
            self.train_nn(self.good_data, self.bad_data)

    def createOutputStep(self):

        if self.predictEnable:
            self.positiveMicrographs = self._createSetOfMicrographs()
            self.positiveMicrographs.setSamplingRate(
                self.imgSet.getSamplingRate())

            for i, img in enumerate(self.imgSet):
                img._probability = Float(self.results[i]['Probability'])

                if self.results[i]['Class'] == 1:
                    self.positiveMicrographs.append(img)

            self._defineOutputs(positiveMicrographs=self.positiveMicrographs)

            self.negativeMicrographs = self._createSetOfMicrographs()
            self.negativeMicrographs.setSamplingRate(
                self.imgSet.getSamplingRate())

            for i, img in enumerate(self.imgSet):
                img._probability = Float(self.results[i]['Probability'])

                if self.results[i]['Class'] == 0:
                    self.negativeMicrographs.append(img)

            self._defineOutputs(negativeMicrographs=self.negativeMicrographs)
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

    def train_nn(self, good_data, bad_data):
        """
        Method to create the model and train it
        """
        trainset = LoaderTrain(good_data, bad_data)
        data_loader = DataLoader(
            trainset, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)
        print('Total data... {}'.format(len(data_loader.dataset)))

        # Set device
        use_cuda = self.useGPU and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', device)
        model = Classify(size_in=(1, 419, 419))
        if self.transferLearning.get():
            model.load_state_dict(torch.load(self.pretrainedModel.get()))
        model = model.to(device)
        print('Model:', model)

        optimizer = optim.Adam(model.parameters(), lr=self.lr.get())
        criterion_train = nn.NLLLoss()
        criterion_test = nn.NLLLoss(size_average=False)

        self.loss_list = []
        self.accuracy_list = []

        for epoch in range(1, self.epochs.get() + 1):
            print('\nEpoch:', epoch, '/', self.epochs.get())
            train(model, device, data_loader, optimizer, criterion_train)
            if self.weightEveryEpoch:
                model.train()
                torch.save(model.cpu(), os.path.join(
                    self.weightFolder.get(), 'model_' + str(epoch) + '.pt'))
                if use_cuda:
                    model.cuda()

            loss, accuracy = self.calcLoss(model, 2, data_loader, device, criterion_test)
            self.loss_list.append(loss)
            self.accuracy_list.append(accuracy)

        if not self.weightEveryEpoch:
            model.train()
            torch.save(model.cpu(), os.path.join(
                self.weightFolder.get(), 'model.pt'))

        print(self.loss_list)
        print(self.accuracy_list)
        self.plot_loss_screening(self.loss_list, self.accuracy_list)

    def plot_loss_screening(self, loss_list, accuracy_list):

        plt.figure(figsize=(9, 7))
        # plt.plot(good, label='Micrografia buena')
        # plt.plot(bad, label='Micrografia mala')
        plt.plot(accuracy_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig('accuracy.png')
                
        plt.figure(figsize=(11, 8))
        plt.plot(loss_list)
        plt.title('Loss function')
        plt.ylabel('Loss function')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss.png')

    def classsify_micrographs(self, images_path):
        """
        Method to prepare the model and classify the micrographs
        """
        trainset = LoaderPredict(images_path)
        data_loader = DataLoader(trainset, batch_size=1,
                                 shuffle=False, num_workers=1, pin_memory=False)
        print('Total data... {}'.format(len(data_loader.dataset)))

        # Set device
        use_cuda = self.useGPU and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', device)
        # weightsfile = "/home/alex/CryoEM/clasifica/model_weight.pt"
        # Create the model and load weights
        model = Classify(size_in=(1, 419, 419))
        model.load_state_dict(torch.load(self.weightsfile.get()))
        model = model.to(device)
        return classify(model, device, data_loader)

    def calcLoss(self, model, n_classes, data_loader, device, loss_function):
        test_loss = 0
        class_correct = [0. for i in range(n_classes)]
        class_total = [0. for i in range(n_classes)]
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                # Move tensors to the configured device
                data, target = data['image'].to(
                    device), data['label'].to(device)
                # Forward pass
                output = model(data)
                # Sum up batch loss
                test_loss += loss_function(output, target).item()
                # Predicted classes
                _, predicted = torch.max(output, 1)
                c = (predicted == target).squeeze()

                # Count classes
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        test_loss /= len(data_loader.dataset)

        # Results of each class
        correct = 0
        for i in range(n_classes):
            correct += class_correct[i]
            print('Accuracy of {}: {}/{} ({:.0f}%)'.format(
                i, int(class_correct[i]), int(class_total[i]),
                100 * class_correct[i] / class_total[i]))
        
        return test_loss, 100. * correct / len(data_loader.dataset)


def classify(model, device, data_loader):
    """
    Method to predict using the neuronal network
    """
    model.eval()
    results = []
    with torch.no_grad():
        for data in data_loader:
            # Move tensors to the configured device
            data = data['image'].to(device)

            # Forward pass
            output = model(data)

            # Predicted class
            probability, predicted = torch.max(output, 1)

            results.append({'Probability': np.exp(
                probability.item()), 'Class': predicted.item()})
    return results


def train(model, device, train_loader, optimizer, loss_function):
    """
    Method to train the neuronal network
    """
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # Move tensors to the configured device
        data, target = data['image'].to(device), data['label'].to(device)

        # Forward pass
        output = model(data)
        loss = loss_function(output, target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print data
        if batch_idx % 20 == 0:
            print('Train: [{}/{} ({:.0f}%)]    \tLoss: {:.6f}'.format(
                batch_idx * train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


class Classify(nn.Module):
    """
    Neuronal Network model
    """

    def __init__(self, size_in=(1, 419, 419), num_classes=2):
        super(Classify, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.flat_size = num_flat_features(self._get_conv_ouput(size_in))
        self.fc1 = nn.Linear(self.flat_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def _get_conv_ouput(self, shape):
        f = torch.rand(1, *shape)
        g = self._forward_conv(f)
        return g

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def num_flat_features(x):
    sizes = x.size()[1:]
    num_features = 1
    for s in sizes:
        num_features *= s
    return num_features


def normalize(mat):
    mean = mat.mean()
    sigma = mat.std()
    mat = (mat - mean) / sigma
    a = mat.min()
    b = mat.max()
    mat = (mat-a)/(b-a)
    return mat


class LoaderPredict(Dataset):
    """
    Class to load the dataset for predict
    """

    def __init__(self, filename_list, size=(1, 419, 419)):
        super(LoaderPredict, self).__init__()
        Plugin.setEnviron()

        self._data = []
        self._size = size
        self._data += [{'file': i} for i in filename_list]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        data = self._data[item]
        img_path = data['file']
        img = loadMrc(img_path)
        img = normalize(img)
        img = np.resize(img, self._size)
        img = torch.from_numpy(img)
        return {'image': img, 'path': img_path}


class LoaderTrain(Dataset):
    """
    Class to load the dataset for train the neuronal network
    """

    def __init__(self, good_data, bad_data, balance=True, size=(1, 419, 419)):
        super(LoaderTrain, self).__init__()
        Plugin.setEnviron()

        self._data = []
        self._size = size

        if(balance):
            n_good = len(good_data)
            n_bad = len(bad_data)
            n_data = n_good if n_good < n_bad else n_bad
            good_data = good_data[:n_data]
            bad_data = bad_data[:n_data]

        self._format_data(good_data, bad_data, self._data)

    def _format_data(self, good_data, bad_data, formated_data):
        formated_data += [{'file': data, 'label': 1}
                          for data in good_data]  # Label 1 is good
        formated_data += [{'file': data, 'label': 0}
                          for data in bad_data]  # Label 0 is bad

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        data = self._data[item]
        img_path = data['file']
        label = data['label']
        img = loadMrc(img_path)
        img = normalize(img)
        img = np.resize(img, self._size)
        img = torch.from_numpy(img)
        return {'image': img, 'label': label}
