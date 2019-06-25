from __future__ import print_function

import pandas as pd
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, genomicData, drugData, res, dgLabel, batch_size=32, shuffle=True, conv1D=False, weight=None):
        # Initialization. Both genomicData and drugData are lists. Each element in the list corresponds to a platform.
        # Each element is a data frame, where rows are samples and columns are features.
        self.genomicData = genomicData
        self.dimGenomicPlat = []
        for i in range(len(self.genomicData)):
            self.dimGenomicPlat.append(self.genomicData[i].shape[1])

        self.drugData = drugData
        self.dimDrugPlat = []
        for i in range(len(self.drugData)):
            self.dimDrugPlat.append(self.drugData[i].shape[1])

        self.batch_size = batch_size
        self.res = res
        self.numSample = res.shape[0]
        self.ids = np.arange(self.numSample)
        self.shuffle = shuffle
        self.dgLabel = dgLabel
        self.numBatch = int(np.floor(self.numSample / self.batch_size))
        self.conv1D = conv1D
        self.weight = weight
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch
        return self.numBatch


    def __getitem__(self, index):
        # Generate one batch of data
#        if self.dgLabel == 'testDG':
#            print(self.dgLabel+'--'+str(index+1)+'/'+str(self.numBatch))
        if index == self.numBatch - 1:
            batchIDs = self.ids[(index * self.batch_size):self.numSample]
        else:
            batchIDs = self.ids[(index * self.batch_size):((index + 1) * self.batch_size)]
        # Initialization
        x = []
        y = np.empty(len(batchIDs))
        for j in range(len(self.genomicData)):
            if not self.conv1D:
                xj = np.empty((len(batchIDs), self.dimGenomicPlat[j]))
                xj[:, :self.dimGenomicPlat[j]] = self.genomicData[j].loc[self.res.iloc[batchIDs]['ccl_name']][:]
            else:
                xj = np.empty((len(batchIDs), self.dimGenomicPlat[j], 1))
                xj[:, :self.dimGenomicPlat[j], 0] = self.genomicData[j].loc[self.res.iloc[batchIDs]['ccl_name']][:]
            x.append(xj)
        for j in range(len(self.drugData)):
            if not self.conv1D:
                xj = np.empty((len(batchIDs), self.dimDrugPlat[j]))
                xj[:, :self.dimDrugPlat[j]] = self.drugData[j].loc[self.res.iloc[batchIDs]['ctrpDrugID']][:]
            else:
                xj = np.empty((len(batchIDs), self.dimDrugPlat[j], 1))
                xj[:, :self.dimDrugPlat[j], 0] = self.drugData[j].loc[self.res.iloc[batchIDs]['ctrpDrugID']][:]
            x.append(xj)
        y[:] = self.res.iloc[batchIDs]['area_under_curve']

        if self.weight is None:
            return x, y
        else:
            return x, y, self.weight.iloc[batchIDs]


    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.ids)


