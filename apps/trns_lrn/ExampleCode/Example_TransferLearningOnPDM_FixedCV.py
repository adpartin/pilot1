from __future__ import print_function

import os
import pandas as pd
import numpy as np
import copy as copy
import shutil
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import _pickle as cp
from scipy import stats

trainStudy = str(sys.argv[1])

# trainStudy = 'CTRP'  # CTRP, GDSC, NCI60, CCLE, gCSI, All

pkl_file = open('../Results_PDM_TransferLearning/LightGBM_CCL_TransferAnalysis_FangfangNewData_IncludePDM_' + trainStudy + \
                '_AUC_dataFilter_False_weight_False_weightFactor_3/AllData.pkl', 'rb')
res = cp.load(pkl_file)
genomics = cp.load(pkl_file)
drug = cp.load(pkl_file)
cclFold = cp.load(pkl_file)
sampleID = cp.load(pkl_file)
para = cp.load(pkl_file)
modelParams = cp.load(pkl_file)
predResult = cp.load(pkl_file)
pkl_file.close()

cclFold = None
sampleID = None
predResult = None

para['numCV'] = 10

# a = para['numFeature']
# para['numFeature'] = {}
# para['numFeature']['numGeneF_50_numDrugF_50'] = a['numGeneF_50_numDrugF_50']
# para['numFeature']['numGeneF_100_numDrugF_50'] = a['numGeneF_100_numDrugF_50']

oriResultFolder = '../Results_PDM_TransferLearning/LightGBM_PDM_FangfangNewData_FixedCV_' + para['trainStudy'] + \
                  '_' + para['measure'] + '_weight_' + str(para['weight']) + '_weightFactor_' + str(para['factor'])
if not os.path.exists(oriResultFolder):
    os.mkdir(oriResultFolder)

pdmID = np.sort(np.where(res.loc[:, 'SOURCE'] == 'PDM')[0])
genomicData = genomics.loc[res.iloc[pdmID, :].loc[:, 'ccl_name'], :]
drugData = drug.loc[res.iloc[pdmID, :].loc[:, 'ctrpDrugID'], :]
x = np.hstack((genomicData.values, drugData.values))
y = res.iloc[pdmID, :].loc[:, 'area_under_curve'].values
groupID = res.iloc[pdmID, :].loc[:, 'groupID'].values
numGroup = int(max(groupID))

if para['trainStudy'] == 'All':
    idTrainStudy = np.where(np.isin(res.loc[:, 'SOURCE'], ['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60']))[0]
else:
    idTrainStudy = np.where(res.loc[:, 'SOURCE'] == para['trainStudy'])[0]

# Transformation of PDM drug response value
z = np.max(np.log2(y)) - np.log2(y)
meanV1 = np.mean(res.iloc[idTrainStudy, :].loc[:, 'area_under_curve'])
stdV1 = np.std(res.iloc[idTrainStudy, :].loc[:, 'area_under_curve'])
meanV2 = np.mean(z)
stdV2 = np.std(z)
z = (z - meanV2) / stdV2 * stdV1 + meanV1

result = {}
for cvID in range(para['numCV']):
    trialFolder = 'trial_' + str(cvID)
    result[trialFolder] = {}

    for foldID in range(para['numFold']):
        cvFolder = 'cv_' + str(foldID)
        result[trialFolder][cvFolder] = {}

        for j in ['AllFeature'] + list(para['numFeature'].keys()):
            params = copy.deepcopy(modelParams[cvFolder][j])
            del params['n_estimators']
            del params['silent']
            del params['importance_type']
            params['objective'] = 'regression'
            params['metric'] = 'l2'

#            params['n_jobs'] = 6

            feature = pd.read_csv('../Results_PDM_TransferLearning/LightGBM_CCL_TransferAnalysis_FangfangNewData_IncludePDM_' + \
                                  para['trainStudy'] + '_AUC_dataFilter_False_weight_False_weightFactor_3/' + cvFolder + \
                                  '/' + j + '_List.txt', sep='\t', header=None, index_col=None, engine = 'c',
                                  na_values = ['na', '-', ''], low_memory=False).values[:, 0]
            predResult = np.empty((len(groupID), 3))
            predResult.fill(np.nan)
            predResult[:, 0] = groupID
            predResult[:, 1] = z

            for i in range(para['numFold']):
                testGroup = pd.read_csv(
                    '../ProcessedData/CV_DataFolds/PDM/' + trialFolder + '/cv_' + str(i) + '/TestList.txt',
                    sep='\t', engine='c', na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
                valGroup = pd.read_csv(
                    '../ProcessedData/CV_DataFolds/PDM/' + trialFolder + '/cv_' + str(i) + '/ValList.txt',
                    sep='\t', engine='c', na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]
                trainGroup = pd.read_csv(
                    '../ProcessedData/CV_DataFolds/PDM/' + trialFolder + '/cv_' + str(i) + '/TrainList.txt',
                    sep='\t', engine='c', na_values=['na', '-', ''], header=None, index_col=None).values[:, 0]

                testID = np.where(np.isin(groupID, testGroup))[0]
                valID = np.where(np.isin(groupID, valGroup))[0]
                trainID = np.where(np.isin(groupID, trainGroup))[0]
                trainData = lgb.Dataset(data=x[trainID, :][:, feature], label=z[trainID])
                validData = lgb.Dataset(data=x[valID, :][:, feature], label=z[valID])

                # modeli = lgb.train(params=params, train_set=trainData, num_boost_round=20, valid_sets=[validData],
                #                    init_model='../Results_PDM_TransferLearning/LightGBM_CCL_TransferAnalysis_FangfangNewData_IncludePDM_' + \
                #                               para['trainStudy'] + '_AUC_dataFilter_False_weight_False_weightFactor_3/' + \
                #                               cvFolder + '/' + j + '_Model', early_stopping_rounds=4, verbose_eval=1)

                modeli = lgb.train(params=params, train_set=trainData, num_boost_round=2000, valid_sets=[validData],
                                   init_model='../Results_PDM_TransferLearning/LightGBM_CCL_TransferAnalysis_FangfangNewData_IncludePDM_' + \
                                              para['trainStudy'] + '_AUC_dataFilter_False_weight_False_weightFactor_3/' + \
                                              cvFolder + '/' + j + '_Model', early_stopping_rounds=300, verbose_eval=1)

                predResult[testID, 2] = modeli.predict(x[testID, :][:, feature])

            predResult = pd.DataFrame(predResult, columns=['GroupID', 'TrueResponse', 'Prediction'])
            sR = copy.deepcopy(predResult)
            keepID = []
            for i in range(numGroup):
                idi = np.where(sR.GroupID == i + 1)[0]
                sR.iloc[idi[0], 1] = np.mean(sR.iloc[idi, :].TrueResponse)
                sR.iloc[idi[0], 2] = np.mean(sR.iloc[idi, :].Prediction)
                keepID.append(idi[0])
            sR = sR.iloc[keepID, :]
            result[trialFolder][cvFolder][j] = sR

analysis = list(result['trial_0']['cv_0'].keys())
summaryTable = np.empty((len(analysis), 7))
summaryTable.fill(np.nan)
index = 0
for j in analysis:
    finalPredResult = np.zeros((numGroup, 3))
    for cvID in range(para['numCV']):
        for foldID in range(para['numFold']):
            finalPredResult = finalPredResult + result['trial_'+str(cvID)]['cv_'+str(foldID)][j].values
    finalPredResult = finalPredResult / para['numFold'] / para['numCV']
    pd.DataFrame(finalPredResult, columns=['GroupID', 'TrueResponse', 'Prediction']).to_csv(oriResultFolder + '/' + j + \
                 '_PDM_PredictionResult.txt', sep='\t', header=True, index=False, line_terminator='\r\n')
    summaryTable[index, 0] = r2_score(finalPredResult[:, 1], finalPredResult[:, 2])
    summaryTable[index, 1] = mean_squared_error(finalPredResult[:, 1], finalPredResult[:, 2])
    summaryTable[index, 2] = mean_absolute_error(finalPredResult[:, 1], finalPredResult[:, 2])
    a, b = stats.pearsonr(finalPredResult[:, 1], finalPredResult[:, 2])
    summaryTable[index, 3] = a
    summaryTable[index, 4] = b
    a, b = stats.spearmanr(finalPredResult[:, 1], finalPredResult[:, 2])
    summaryTable[index, 5] = a
    summaryTable[index, 6] = b
    index = index + 1

pd.DataFrame(summaryTable, columns=['r2', 'mse', 'mae', 'pearson-cor', 'pearson-pvalue', 'spearman-cor', 'spearman-pvalue'],
             index=analysis).to_csv(oriResultFolder + '/PDM_PredictionSummary.txt', sep='\t', header=True, index=True,
             line_terminator='\r\n')

output = open(oriResultFolder + '/AllPredictionResult.pkl', 'wb')
cp.dump(result, output)
output.close()