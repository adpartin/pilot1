from __future__ import print_function

import os
import pandas as pd
import numpy as np
import copy as copy
import shutil
import sys
from lightGBM_FS_CCL_TransferAnalysis import lightGBM
from loadData import loadDataLightGBM_PDM
import _pickle as cp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score

para = {}

para['trainStudy'] = str(sys.argv[1])
para['filterFlag'] = int(sys.argv[2])

# para['trainStudy'] = 'CCLE' # trainStudy can be CTRP, GDSC, NCI60, CCLE, gCSI, All
# para['filterFlag'] = int(1)

if para['filterFlag'] == 0:
    para['filterFlag'] = False
else:
    para['filterFlag'] = True

para['factor'] = int(3)

temp = [[50, 50], [100, 50], [50, 100], [100, 100], [200, 100], [100, 200], [200, 200], [200, 400], [400, 200], [400, 400],
      [400, 800], [800, 400], [800, 800], [800, 1600], [1600, 800], [1600, 1600]]
# temp = [[50, 50], [100, 100]]

para['numFeature'] = {}
for i in range(len(temp)):
    s = 'numGeneF_'+str(temp[i][0])+'_numDrugF_'+str(temp[i][1])
    para['numFeature'][s] = temp[i]

para['measure'] = 'AUC'
para['drT1'] = 0.5
para['weight_drT'] = 'drT1'
para['weight'] = False
para['numFold'] = 10

para['n_estimators'] = 2000
# para['n_estimators'] = 20

para['n_jobs'] = 30
# para['n_jobs'] = 6

para['random_state'] = 1

para['early_stopping_rounds'] = 400
# para['early_stopping_rounds'] = 4

para['num_leaves'] = 31

para['folder'] = 'LightGBM_CCL_TransferAnalysis_FangfangNewData_IncludePDM_' + para['trainStudy'] + '_' + para['measure'] + \
                 '_dataFilter_' + str(para['filterFlag']) + '_weight_' + str(para['weight']) + '_weightFactor_' + str(para['factor'])

oriResultFolder = '../Results_PDM_TransferLearning/' + para['folder']
if not os.path.exists(oriResultFolder):
    os.mkdir(oriResultFolder)

if para['filterFlag']:
    res = pd.read_csv('../ProcessedData/DrugResponse/combined_single_response_agg_filtered_unique_ccl_drug_filteredByR2fitEC50se.txt',
                      sep='\t', header=0, engine = 'c', na_values = ['na', '-', ''], low_memory=False)
else:
    res = pd.read_csv('../ProcessedData/DrugResponse/combined_single_response_agg_filtered_unique_ccl_drug.txt',
                      sep='\t', header=0, engine = 'c', na_values = ['na', '-', ''], low_memory=False)

col = [str(i) for i in res.columns]
col[1] = 'ccl_name'
col[2] = 'ctrpDrugID'
col[9] = 'groupID'
id = np.where(np.array(col) == para['measure'])[0]
col[id[0]] = 'area_under_curve'
res.columns = col
res = res.iloc[:, [0, 1, 2, id[0], 9]]

res2 = pd.read_csv('../ProcessedData/PDM_Data/SingleDrugPdxDataFrame_2019-01-31.txt', sep='\t', header=0, engine = 'c', na_values = ['na', '-', ''], low_memory=False)
res3 = copy.deepcopy(res.iloc[:res2.shape[0], :])
res3.SOURCE = 'PDM'
res3.ccl_name = res2.UniqueCCL_ID
res3.ctrpDrugID = res2.UniqueDrugID
res3.groupID = res2.groupID
res3.area_under_curve = res2.loc[:, 'RM-EFS']
res3 = res3.loc[:, ['SOURCE', 'ccl_name', 'ctrpDrugID', 'area_under_curve', 'groupID']]
res = pd.concat(objs=(res, res3), axis=0)

print(str(np.sum(np.isnan(res.iloc[:, 3]))))
print(str(res.shape[0]) + '_' + str(res.shape[1]))

id = np.where(np.invert(np.isnan(res.iloc[:, 3])))[0]
res = res.iloc[id, :]

print(str(np.sum(np.isnan(res.iloc[:, 3]))))
print(str(res.shape[0]) + '_' + str(res.shape[1]))


####################################
# randID = np.random.permutation(res.shape[0])
# res = res.iloc[randID[:50000], :]

if para['trainStudy'] == 'All':
    idTrainStudy = np.where(np.isin(res.loc[:, 'SOURCE'], ['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60']))[0]
    para['testStudy'] = []
else:
    idTrainStudy = np.where(res.loc[:, 'SOURCE'] == para['trainStudy'])[0]
    para['testStudy'] = list(set(['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60']).difference(set([para['trainStudy']])))

weight = pd.DataFrame(np.ones((res.shape[0], 1)), index=res.index, columns=['weight'])
id_pos = np.where(res.iloc[idTrainStudy, :].loc[:, 'area_under_curve'] < para[para['weight_drT']])[0]
id_neg = np.where(res.iloc[idTrainStudy, :].loc[:, 'area_under_curve'] >= para[para['weight_drT']])[0]
if para['measure'] == 'DSS1':
    weight.iloc[idTrainStudy[id_neg], 0] = weight.iloc[idTrainStudy[id_neg], 0] * para['factor'] * len(id_pos) / len(id_neg)
else:
    weight.iloc[idTrainStudy[id_pos], 0] = weight.iloc[idTrainStudy[id_pos], 0] * para['factor'] * len(id_neg) / len(id_pos)
res = pd.concat(objs=(res, weight), axis=1)
numS = res.shape[0]

genomics, drug = loadDataLightGBM_PDM(ccl=np.unique(res.ccl_name), drug=np.unique(res.ctrpDrugID))

allCCL = np.unique(res.iloc[idTrainStudy, :].loc[:, 'ccl_name'])
randID = np.random.permutation(len(allCCL))
allCCL = allCCL[randID]

foldSize = np.ceil(len(allCCL)/para['numFold'])
cclFold = []
for i in range(para['numFold']):
    startIndex = int(i*foldSize)
    endIndex = int(min((i+1)*foldSize, len(allCCL)))
    cclFold.append(list(allCCL[startIndex:endIndex]))

sampleID = {}
predResult = {}
modelParams = {}
for foldID in range(para['numFold']):
    cvFolder = 'cv_'+str(foldID)
    print('****************************************************************')
    print(cvFolder)

    resultFolder = oriResultFolder+'/'+cvFolder
    if os.path.exists(resultFolder):
        shutil.rmtree(resultFolder)
    os.mkdir(resultFolder)

    testCCL = cclFold[foldID]
    if foldID < para['numFold'] - 1:
        valCCL = cclFold[foldID+1]
    else:
        valCCL = cclFold[0]
    trainCCL = list(np.setdiff1d(allCCL, np.union1d(valCCL, testCCL)))

    sampleID[cvFolder] = {}
    sampleID[cvFolder]['trainID'] = idTrainStudy[np.where(np.isin(res.iloc[idTrainStudy, :].loc[:, 'ccl_name'], trainCCL))[0]]
    sampleID[cvFolder]['valID'] = idTrainStudy[np.where(np.isin(res.iloc[idTrainStudy, :].loc[:, 'ccl_name'], valCCL))[0]]
    sampleID[cvFolder]['testID'] = {}
    sampleID[cvFolder]['testID']['testID'] = np.sort(idTrainStudy[np.where(np.isin(res.iloc[idTrainStudy, :].loc[:, 'ccl_name'], testCCL))[0]])
    if len(para['testStudy']) > 0:
        for s in para['testStudy']:
            sampleID[cvFolder]['testID'][s] = np.sort(np.where(res.loc[:, 'SOURCE'] == s)[0])

    params, pred = lightGBM(res=res, sampleID=sampleID[cvFolder], genomics=genomics, drug=drug, resultFolder=resultFolder, para=para,
                      numFeature=para['numFeature'], weight=para['weight'])
    modelParams[cvFolder] = params
    predResult[cvFolder] = pred

output = open(oriResultFolder + '/AllData.pkl', 'wb')
cp.dump(res, output)
cp.dump(genomics, output)
cp.dump(drug, output)
cp.dump(cclFold, output)
cp.dump(sampleID, output)
cp.dump(para, output)
cp.dump(modelParams, output)
cp.dump(predResult, output)
output.close()

analysis = sorted(list(predResult['cv_0'].keys()))
analysis1 = np.array([i.split(sep='-')[0] for i in analysis])
analysis2 = np.array([i.split(sep='-')[1] for i in analysis])
summaryTable = np.empty((len(analysis), 7))
summaryTable.fill(np.nan)
index = 0
for j in range(len(analysis)):
    if analysis2[j] == 'testID':
        result = copy.deepcopy(predResult['cv_0'][analysis[j]])
        for foldID in range(1, para['numFold']):
            result = pd.concat(objs=(result, predResult['cv_' + str(foldID)][analysis[j]]), axis=0)
        result.to_csv(oriResultFolder + '/' + analysis[j] + '_PredictionResult.txt', sep='\t', header=True, index=False,
                      line_terminator='\r\n')
    else:
        result = copy.deepcopy(predResult['cv_0'][analysis[j]])
        sep = pd.DataFrame(['|' for i in range(result.shape[0])], columns=['sep'], index=result.index)
        for foldID in range(1, para['numFold']):
            if np.sum(result.SOURCE + sep.sep + result.ccl_name + sep.sep + result.ctrpDrugID != predResult['cv_' + str(foldID)][analysis[j]].SOURCE +
                      sep.sep + predResult['cv_' + str(foldID)][analysis[j]].ccl_name + sep.sep + predResult['cv_' + str(foldID)][analysis[j]].ctrpDrugID) > 0:
                sys.exit('Sample order in result not matched for ' + analysis[j])
            else:
                result.prediction = result.prediction + predResult['cv_' + str(foldID)][analysis[j]].prediction
        result.prediction = result.prediction / para['numFold']
        result.to_csv(oriResultFolder + '/' + analysis[j] + '_PredictionResult.txt', sep='\t', header=True, index=False,
                      line_terminator='\r\n')

    summaryTable[index, 0] = r2_score(result.loc[:, 'area_under_curve'], result.loc[:, 'prediction'])
    summaryTable[index, 1] = mean_squared_error(result.loc[:, 'area_under_curve'], result.loc[:, 'prediction'])
    summaryTable[index, 2] = mean_absolute_error(result.loc[:, 'area_under_curve'], result.loc[:, 'prediction'])
    summaryTable[index, 3] = roc_auc_score(result.loc[:, 'area_under_curve'] >= para['drT1'], result.loc[:, 'prediction'])
    summaryTable[index, 4] = accuracy_score(result.loc[:, 'area_under_curve'] >= para['drT1'], result.loc[:, 'prediction'] >= para['drT1'])
    summaryTable[index, 5] = np.sum(np.multiply(result.loc[:, 'area_under_curve'] >= para['drT1'], result.loc[:, 'prediction'] >= para['drT1'])) / np.sum(result.loc[:, 'area_under_curve'] >= para['drT1'])
    summaryTable[index, 6] = np.sum(np.multiply(result.loc[:, 'area_under_curve'] < para['drT1'], result.loc[:, 'prediction'] < para['drT1'])) / np.sum(result.loc[:, 'area_under_curve'] < para['drT1'])
    index = index + 1

summaryTable = pd.DataFrame(summaryTable, columns=['r2', 'mse', 'mae', 'auroc', 'accuracy', 'TNR', 'TPR'])
analysis1 = pd.DataFrame(analysis1.reshape(len(analysis1), 1), columns=['feature'], index=summaryTable.index)
analysis2 = pd.DataFrame(analysis2.reshape(len(analysis2), 1), columns=['testData'], index=summaryTable.index)
summaryTable = pd.concat(objs=(analysis1, analysis2, summaryTable), axis=1)
summaryTable.to_csv(oriResultFolder + '/PredictionResultSummary.txt', sep='\t', header=True, index=False, line_terminator='\r\n')
