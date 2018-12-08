from __future__ import print_function
import pandas as pd
import numpy as np
#import cPickle as pickle
import pickle 
import matplotlib.pyplot as plt
from collections import Counter
from dataPrepFunctions import binDataByQuartiles
'''
from src.dataPrepFunctions import binDataByQuartiles
'''
import tqdm as tqdm
from sklearn.preprocessing import LabelEncoder
from time import time 

fraction = 0.5  ### 95% complete -> 0.05;   50% complte -> 0.5
count_threshold = 10 
dataFolder = '/Users/futianfan/Downloads/Gatech_Courses/emory_kidney/data'
trainData = pd.read_csv('{}/deidData_GFtrainData_Nov2018.csv'.format(dataFolder))
testData = pd.read_csv('{}/deidData_GFtestData_Nov2018.csv'.format(dataFolder))
Num_train = trainData.shape[0]
trainData = pd.concat([trainData, testData])
Ntrain = trainData.shape[0]

# Drop columns with 0 variance
trainColsToDrop = trainData.columns[trainData.nunique() == 1].values
trainData.drop(trainColsToDrop,axis = 'columns',inplace = True)

stateVariables = ['perm_state_l_22','dstate_22','state_residenc','perm_state_22']
droplist = ['usrds_id','gf_6months','gf_3year'] + ['perm_state_l_22','dstate_22','state_residenc','perm_state_22'] \
+ ['unosstat_waitlist_ki']  + ['rsex_22'] \
 + ['matc_22'] ## + ['first_mcare_pta_reason_patients', 'first_mcare_pta_status_patients']
## ['don_hiv_22'] 

trainData.drop(droplist, axis = 'columns', inplace = True)

# Create feature lists
nonFeatures = ['gf_1year']
features = [x for x in trainData.columns ]
trainFeature = trainData[features ]

### 1. 95% completeness
complete_filter = lambda x: trainFeature[x].isna().sum() < fraction * len(trainFeature[x])
features_complete = list(filter(complete_filter, features))
assert 'don_hiv_22' in features_complete
trainFeature = trainFeature[features_complete]
print('after 95 completeness, feature dim is {}'.format(len(features_complete)))

### 2.  TYPE filter: detect categoricalVariables and nonCategoricalVariables.
nonCategoricalVariables = list(trainFeature.columns[trainFeature.dtypes == float])
print('1. noncategorical num is {}'.format(len(nonCategoricalVariables)))      ### 59
categoricalVariables = list(trainFeature.columns[trainFeature.dtypes == object])
print('1. categorical num is {}'.format(len(categoricalVariables)))            ### 45
  ### throw away the column that has too many categories  count_threshold 
'''
trainColsToDrop = trainFeature.columns[trainFeature.nunique() > count_threshold].values
trainColsToDrop = set(trainColsToDrop)
trainColsToDrop = list(trainColsToDrop.intersection(set(categoricalVariables)))
trainColsToDrop = np.array(trainColsToDrop)
trainFeature.drop(trainColsToDrop,axis = 'columns',inplace = True)
'''  ### throw away the column that has too many categories  count_threshold


### compute num of categorical and noncategorical features
nonCategoricalVariables = list(trainFeature.columns[trainFeature.dtypes == float])
print('2. noncategorical num is {}'.format(len(nonCategoricalVariables)))      ### 59
categoricalVariables = list(trainFeature.columns[trainFeature.dtypes == object])
print('2. categorical num is {}'.format(len(categoricalVariables)))            ### 45
assert 'acute_rej_epi_22' in nonCategoricalVariables


### (a) file with NA for categorical Variables 
trainFeature[categoricalVariables] = trainFeature[categoricalVariables].fillna("NA")
### (b) file with average value for noncategorical Variables
for feat in nonCategoricalVariables: ## nonCategoricalVariables, ['acute_rej_epi_22'] 
  avg = float(trainFeature[feat].mode())
  trainFeature[feat] = trainFeature[feat].fillna(avg)


assert trainFeature[nonCategoricalVariables].isna().sum().sum() == 0
trainFeature, feature_map = binDataByQuartiles(trainFeature, nonCategoricalVariables)
assert trainFeature[nonCategoricalVariables].isna().sum().sum() == 0


nonCategoricalVariables = list(trainFeature.columns[trainFeature.dtypes == "category"])
print('3. noncategorical num is {}'.format(len(nonCategoricalVariables)))      ### 59
categoricalVariables = list(trainFeature.columns[trainFeature.dtypes == object])
print('3. categorical num is {}'.format(len(categoricalVariables)))            ### 45



def PdFrame2FileLine(DataFrame, categoricalVariables, nonCategoricalVariables, nonFeatures, separate_symbol = ';'):
  #DataFrame = trainFeature, testFeature
  N = DataFrame.shape[0]
  feat_name = []
  #total_line = []
  num_feat = len(nonFeatures) + len(nonCategoricalVariables)
  data = np.zeros((N,num_feat), dtype = float)
  ### nonFeatures
  for i,feat in enumerate(nonFeatures):
    dataColumn = DataFrame[feat].to_string()
    #assert separate_symbol not in dataColumn
    dataColumn = [float(i.split()[1]) for i in dataColumn.split('\n')]
    dataColumn = np.array(dataColumn)
    data[:,i] = dataColumn
    feat_name.append(feat)
  ###  nonCategoricalVariables
  for i,feat in enumerate(nonCategoricalVariables):
    dataColumn = DataFrame[feat].to_string()
    dataColumn = [float(i.split()[1]) for i in dataColumn.split('\n')[:-1]]
    dataColumn = np.array(dataColumn)
    data[:, len(nonFeatures) + i] = dataColumn
    feat_name.append(feat)
  ###  categoricalVariables
  for feat in categoricalVariables:
    dataColumn = DataFrame[feat].to_string()
    dataColumn = [i.split()[1] for i in dataColumn.split('\n')]
    value_set = list(set(dataColumn))
    leng = len(value_set)
    num_feat += leng
    print('set length is {}'.format(leng))
    value_indx = lambda x: value_set.index(x)
    dataColumn = list(map(value_indx, dataColumn))
    def create_one_hot_vector(indx):
      vec = [0] * leng
      vec[indx] = 1
      return vec
    datamat = list(map(create_one_hot_vector, dataColumn))
    datamat = np.array(datamat, dtype = float)
    assert N, leng == datamat.shape
    #feat_name.extend([feat] * leng)
    for j in range(leng):
      feat_name.append(feat + "__is__" + value_set[j])
    data = np.concatenate((data, datamat), 1)
  return data, feat_name

t1 = time()
data, feat_name = PdFrame2FileLine(trainFeature, categoricalVariables, nonCategoricalVariables, nonFeatures, separate_symbol = ';')
print('cost {} seconds'.format(time() - t1))

np.save('{}/train.npy'.format(dataFolder), data[:Num_train,:])
np.save('{}/test.npy'.format(dataFolder), data[Num_train:,:])
with open('{}/feature_name_for_all_data'.format(dataFolder), 'w') as fout:
  for i in feat_name:
    fout.write(i + '\t')

with open('{}/feature_map'.format(dataFolder), 'w') as fout:
  for i in feature_map:
    fout.write(i + '\t' + feature_map[i] + '\n')



"""

# Non Categorical Variables identified manually
# nonCategoricalVariables = ['age_22','inc_age_patients',
#                        'los_22','dcreat_22',
#                        'CENSUS_UnemploymentRate','CENSUS_PercentInsured','CENSUS_PercentFamiliesBelowPoverty',
#                        'CENSUS_percentAA','dage_22','mrcreat_22','dwgt_22','wl_time_22',
#                        'dialysis_vintage_22','time_esrd_tx_patients']
nonCategoricalVariables = ['CENSUS_PercentFamiliesBelowPoverty',
 'CENSUS_PercentInsured',
 'CENSUS_UnemploymentRate',
 'CENSUS_percentAA',
 'age_22',
 'dage_22',
 'dcreat_22',
 'dialysis_vintage_22',
 'dwgt_22',
 'inc_age_patients',
 'los_22',
 'mis_matc_22',
 'mrcreat_22',
 'provusrd_tx_22',
 'time_esrd_tx_patients',
 'transplant_year',
 'wl_time_22']
#### length 17 
#### 17 -> 13 
nonCategoricalVariables = [x for x in nonCategoricalVariables if x in features]
#### length 13
# The rest must be categorical
categoricalVariables = [x for x in features if x not in nonCategoricalVariables]

print(len(nonCategoricalVariables))
print(len(categoricalVariables))

print(trainData[categoricalVariables].isna().sum().sum())
trainData[categoricalVariables] = trainData[categoricalVariables].fillna("NA")
print(trainData[categoricalVariables].isna().sum().sum())

print(testData[categoricalVariables].isna().sum().sum())
testData[categoricalVariables]= testData[categoricalVariables].fillna("NA")
print(testData[categoricalVariables].isna().sum().sum())

# Bin data into low-med-high = 0-1-2
trainData = binDataByQuartiles(trainData,nonCategoricalVariables)
testData = binDataByQuartiles(testData,nonCategoricalVariables)
print(trainData.shape)

'''
### CHECK
# Check that all columns are accounted for
if(len(trainData.columns) - len(nonCategoricalVariables) - len(categoricalVariables) - len(nonFeatures) == 0):
    print("The number of categorical and non-categorical columns add up")
else:
    raise ValueError("The number of categorical and non-categorical columns don't add up")
## Make sure no null values
if (trainData[nonCategoricalVariables].isna().sum().sum() == 0):
    print("No Null Values")
else:
    raise ValueError('Null values introduced after binning!')
## Make sure no null values
if (trainData[nonCategoricalVariables].isna().sum().sum() == 0):
    print( "No Null Values")
else:
    raise ValueError('Null values introduced after binning!')
'''
### CHECK
assert len(trainData.columns) - len(nonCategoricalVariables) - len(categoricalVariables) - len(nonFeatures) == 0
assert trainData[nonCategoricalVariables].isna().sum().sum() == 0
assert testData[nonCategoricalVariables].isna().sum().sum() == 0
### CHECK

# Check the number of categories
print("Categorical")
print( trainData[categoricalVariables].nunique().describe() )
print("\n Non-categorical")
print( trainData[nonCategoricalVariables].nunique().describe() )


trainData.to_csv('{}/train_gf_1yr.csv'.format(dataFolder),index = False)
testData.to_csv('{}/test_gf_1yr.csv'.format(dataFolder),index = False)




'''
combinedFeatures = trainData[features].append(testData[features])
print(combinedFeatures.shape)
le = LabelEncoder()
le.fit(combinedFeatures.iloc[:,1])
for col in features:
    le.fit(combinedFeatures[col])
    trainData[col] = le.transform(trainData[col])
    testData[col] = le.transform(testData[col])
'''


'''
trainData[categoricalVariables].nunique().describe()
trainData[nonCategoricalVariables].nunique().describe()

for col in trainData.columns:
    if trainData.loc[:,col].dtypes != 'int64':
        raise ValueError('Non-integer column {} found!'.format(col))

for col in testData.columns:
    if testData.loc[:,col].dtypes != 'int64':
        raise ValueError('Non-integer column {} found!'.format(col)) 
'''




def find_setDiff(trainData,testData,features):
    '''
    Returns a list of list containing [columName,Extra Categories present in testData that aren't in trainData]
    '''
    setdiffList = list()
    for col in features:
        setdiff = np.setdiff1d(np.unique(testData.loc[:,col]),np.unique(trainData.loc[:,col]))
        if len(setdiff) > 0:
            setdiffList.append([col,setdiff])  
    return setdiffList


setDiffList = find_setDiff(trainData,testData,features)




# For these columns, find the testData rows which have extra category levels
print(trainData.shape)
print(testData.shape)

for setDiff in setDiffList:
    rowToMove = testData.loc[testData[setDiff[0]].isin(setDiff[1]),:].index
    trainData = trainData.append(testData.iloc[rowToMove,:])
    testData = testData.drop(rowToMove)
print(trainData.shape)
print(testData.shape)


# Check the distribution of he features after the conversion
print( 100.0*trainData['gf_1year'].sum()/len(trainData) )
print( 100.0*testData['gf_1year'].sum()/len(testData) )


import pandas as pd
import numpy as np

bithVars = np.unique(['rabo_22',
                      'rhisp_22',
                      'rrace_22',
                      'sex_patients'])
listingVars = np.unique(['diabr_22',
                         'educ_22',
                         'exvasacc_22',
                         'funcstl_22',
                         'pripay_l_22',
                         'prmalr_22',
                         'pvascr_22',
                         'rcitz_22',
                         'trcopdr_22',
                         'trdgn_l_22',
                        'pdis_patients'])
pretxVars = np.unique(['cmvigg_22',
                       'cmvigm_22',
                       'dabo_22',
                       'dhistcig_22',
                       'dial_change_pretx_rxhist',
                       'disgrpc_patients',
                       'malig_22',
                       'recip_phys_capacity_change_22',
                      'sera_test_class1_22',
                       'sera_test_class2_22',
                       'inc_age_patients',
                      'wl_time_22',
                       'dialysis_vintage_22',
                       'time_esrd_tx_patients',
                      'trdgn_22'])
txVars = np.unique(['abo_match_22',
                    'cmv_react_risk_22',
                    'funcstat_22',
                    'hist_cancer_22', 
                    'kpproc_22',
                    'hcv_serostatus_22',
                    'mdcond_22',
                    'network_residenc', 
                    'org_rec_on_22',
                    'organ_recover_time_22',
                    'pripay_22',
                    'work_income_22',
                    'xmat_oth_ser_22',
                    'mis_matc_22',
                    'transplant_year',
                   'pre_tx_txfus_22',
                    'age_22'])
postTxVars = np.unique(['acute_rej_epi_22',
                        'fwdial_22',
                        'organ_recover_time_22',
                        'los_22',
                       'mrcreat_22'])


print(len(trainData[bithVars].columns))
print(len(trainData[listingVars].columns))
print(len(trainData[pretxVars].columns))
print(len(trainData[txVars].columns))
print(len(trainData[postTxVars].columns))

temporalTiers = [bithVars,listingVars,pretxVars,txVars,postTxVars]
flattenedTemporalTiers = [item for sublist in temporalTiers for item in sublist]



## DONOR/RECIPIENT TIERS
# We want to exclude any arrows from donor variables to pre-tx variables for RECIPIENTS

# Create donor/recipient tiers
recipientVars = np.unique(['acute_rej_epi_22',
                           'cmvigg_22',
                           'cmvigm_22',
                           'diabr_22',
                           'dial_change_pretx_rxhist',
                           'disgrpc_patients',
                           'dualelig_payhist',
                           'educ_22',
                           'exvasacc_22',
                           'first_mcare_pta_reason_patients',
                           'first_mcare_pta_status_patients',
                           'first_mcare_ptb_reason_patients',
                           'first_mcare_ptb_status_patients',
                           'first_modality_patients',
                           'funcstat_22',
                           'funcstl_22',
                           'fwdial_22',
                           'hcv_serostatus_22',
                           'hist_cancer_22',
                           'kpproc_22',
                           'malig_22',
                           'mdcond_22',
                           'network_residenc',
                           'org_rec_on_22',
                           'organ_recover_time_22',
                           'pre_tx_txfus_22',
                           'pripay_22',
                           'pripay_l_22',
                           'prmalr_22',
                           'pvascr_22',
                           'rabo_22',
                           'rcitz_22',
                           'recip_phys_capacity_change_22',
                           'rhisp_22',
                           'rrace_22',
                           'sera_test_class1_22',
                           'sera_test_class2_22',
                           'sex_patients',
                           'trcopdr_22',
                           'usa_residenc',
                           'work_income_22',
                           'transplant_year',
                           'age_22',
                           'trdgn_l_22',
                           'trdgn_22',
                           'inc_age_patients',
                           'los_22',
                           'pdis_patients',
                           'provusrd_tx_22',
                           'mrcreat_22',
                           'wl_time_22',
                           'dialysis_vintage_22',
                           'time_esrd_tx_patients'])

# Create a list of RECIPIENT pre-tx temporal vars
pretx_temporalTiers = [bithVars,listingVars,pretxVars]
pretx_flattenedTemporalTiers = [item for sublist in pretx_temporalTiers for item in sublist]

pretxRecipient = [x for x in recipientVars if x not in pretx_flattenedTemporalTiers]

# List the donor vars
donorVars = np.unique(['dabo_22',
                       'dcitz_22',
                       'dhhyp_22',
                       'dhisp_22',
                       'dhistcig_22',
                       'diab_don_22',
                       'don_hbv_22',
                       'don_hcv_22',
                       'hiv_don_22',
                       'dcreat_22',
                       'dage_22',
                       'dwgt_22',
                       'don_physical_capacity_22',
                       'don_urine_protein_22',
                       'donrel_22',
                       'drace_22',
                       'dsex_22',
                       'dtype_22'])


print(len(trainData[recipientVars].columns))
print(len(trainData[donorVars].columns))
print(len(trainData[pretxRecipient].columns))



# Create Blacklist dataframe
labels =  ['gf_1year']
features = [x for x in trainData.columns if x not in labels]

blacklist = pd.DataFrame(columns = ['from','to'])

# No arrows from label to features
for label in labels:
    for feature in features:
        blacklist = blacklist.append({'from':label,'to':feature},ignore_index = True)

# Temoral Tiers
for tier in temporalTiers:
    otherTiers = [x for x in flattenedTemporalTiers if x not in tier]
    
    for var in tier:
        for otherVar in otherTiers:
            blacklist = blacklist.append({'from':otherVar,'to':var},ignore_index = True)
            
# Donor/Recipient Tiers
for donVar in donorVars:
    for pretxRecipVar in pretxRecipient:
        blacklist = blacklist.append({'from':donVar,'to':pretxRecipVar},ignore_index = True)


# Check that the blacklist does not have any extra nodes
fromVars = np.unique(blacklist['from'])
toVars = np.unique(blacklist['to'])

print( len(trainData[fromVars].columns))
print( len(trainData[toVars].columns))


# Check that the blacklist does not have any extra nodes
fromVars = np.unique(blacklist['from'])
toVars = np.unique(blacklist['to'])

print( len(trainData[fromVars].columns) )
print( len(trainData[toVars].columns) )


# 1 year graft failure dataset
trainData_1yr = trainData.copy()
testData_1yr = testData.copy()

trainData_1yr.to_csv('/project/emory/CausalDiscovery/bnlearnFiles/trainData_1yr.csv',index = False)
testData_1yr.to_csv('/project/emory/CausalDiscovery/bnlearnFiles/testData_1yr.csv',index = False)


blacklist.to_csv('/project/emory/CausalDiscovery/bnlearnFiles/blacklist_constrained.csv',index = False)


# 1 year graft failure dataset

trainData_1yr = trainData.copy()
testData_1yr = testData.copy()

# # Remove the other 2 gf labels
# trainData_1yr.drop(['gf_6months','gf_3year'],axis = 'columns',inplace = True)
# testData_1yr.drop(['gf_6months','gf_3year'],axis = 'columns',inplace = True)


trainData_1yr.to_csv('/project/emory/CausalDiscovery/tetradFiles/run1/trainData_1yr.csv',index = False)
testData_1yr.to_csv('/project/emory/CausalDiscovery/tetradFiles/run1/testData_1yr.csv',index = False)



labels =  ['gf_1year']
features = [x for x in trainData.columns if x not in labels]

with open('/project/emory/CausalDiscovery/tetradFiles/run1/knowledge_constrained.prior','wb') as f:
    f.write('/knowledge\n')
    
    f.write('forbiddirect\n')
    
    for row in blacklist.values:
        f.write('{} {}'.format(row[0],row[1]))
        f.write('\n')
    
    f.write('addtemporal\n')
    f.write('1 ')
    for x in features:
        f.write(x)
        f.write(' ')
    
    f.write('\n')
    
    f.write('2 ')
    for x in labels:
        f.write(x)
        f.write(' ')        
        
    f.write('\n')


"""
















