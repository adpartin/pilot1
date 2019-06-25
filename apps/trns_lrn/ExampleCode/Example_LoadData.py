import _pickle as cp

pkl_file = open('../Data/CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl', 'rb')
res = cp.load(pkl_file)
genomics = cp.load(pkl_file)
drug = cp.load(pkl_file)
pkl_file.close()
print('Done.')
