import pandas as pd
import numpy as np
from pathos import multiprocessing as mp
from tqdm import tqdm
import scipy.stats
import datetime
import itertools



def binDataByQuartiles(df,colsToBin):
    '''
    Returns a quartile-binned version of inputData
    '''
    data = df.reset_index(drop = True).copy()
    
    colsDontBin = [x for x in data.columns if x not in colsToBin]
    
    binData = data[colsToBin].copy()
    colNames = binData.columns
    
    # Calculate the quartile information for each column
    colMins = binData.min()
    colQ1s = binData.quantile(0.25)
    colQ2s= binData.quantile(0.5)
    colQ3s = binData.quantile(0.75)
    colMaxs = binData.max()
    label3 = [-1, 0, 1]
    label2 = [-1, 1]
    
    feature_map = dict()
    
    for i in range(len(colNames)):
        var_name = colNames[i]
        
        ##print('Variable : {}'.format(var_name))
        
        minVal = colMins[i]
        lowQVal = colQ1s[i]
        medVal = colQ2s[i]
        highQVal = colQ3s[i]
        maxVal = colMaxs[i]

        tol = 1e-3
        maxVal += tol
        minVal -= tol 

        binEdges = [minVal,lowQVal,highQVal,maxVal]
        #print(binEdges)
        #print('\t binEdges: {}'.format(binEdges))
        #print('\t Number of unique bin edges: {}'.format(len(set(binEdges))))
        
        
        if len(set(binEdges)) == 4: #check how many unique bin edges
            # Split into 3 bins: Low, Med, and High
            #print( "\t Binning variable {} with binEdges {}\n".format(var_name,binEdges), )
            binData[var_name] = pd.cut(binData[var_name],binEdges,
                labels=label3, include_lowest = True) ## labels=[0,1,2]
            feature_map[var_name] = str(lowQVal) + ":" + str(highQVal)
        elif len(set(binEdges)) == 3:
            #raise ValueError('Less than 4 unique bin edges. Double check the function for this case')
            if(binEdges[0] == binEdges[1]):
                #Box skewed to the left: minVal = lowQVal
                binEdges = [minVal,medVal,highQVal,maxVal]
                if(len(set(binEdges))) == 3:
#                     #minVal is equal to medVal, so new binEdges [minVal, highQVal, maxVal]
#                     #split into top and bottom
                    binEdges = [minVal,highQVal,maxVal]
#     #                 print "Binning variable {} with binEdges {}\r".format(var_name,binEdges),
                    binData[var_name] = pd.cut(binData[var_name],binEdges,
                        labels=label2)  ### labels=['bottom','top']
                    feature_map[var] = str(highQVal)
                else:
#                     #minVal is not equal to medVal, so use binEdges as is
#     #                 print "Binning variable {} with binEdges {}\r".format(var_name,binEdges),
                    binData[var_name] = pd.cut(binData[var_name],binEdges,
                        labels=label3)  ### labels=['low','med','high']
                    feature_map[var_name] = str(medVal) + ":" + str(highQVal)

            elif(binEdges[2] == binEdges[3]):
#                 #Box skewed to the right: highQVal = maxVal
                binEdges = [minVal,lowQVal,medVal,maxVal]
                if(len(set(binEdges))) == 3:
#                     #maxVal is equal to medVal, so new binEdges [minVal, lowQVal,maxVal]
#                     #split into top and bottom
                    binEdges = [minVal,lowQVal,maxVal]
#     #                 print "Binning variable {} with binEdges {}\r".format(var_name,binEdges),
                    binData[var_name] = pd.cut(binData[var_name],binEdges,
                        labels=label2)   #### labels=['bottom','top']
                    feature_map[var_name] = str(lowQVal)
                else:
#                     #maxVal is not equal to medVal, so use binEdges as is
#     #                 print "Binning variable {} with binEdges {}\r".format(var_name,binEdges),
                    binData[var_name] = pd.cut(binData[var_name],binEdges,
                        labels=label3)   ####  labels=['low','med','high']
                    feature_map[var_name] = str(lowQVal) + ":" + str(medVal)
            elif(binEdges[1] == binEdges[2]):
#                     #lowQVal = medval = highQVal
                    binEdges = [minVal,lowQVal,maxVal]
#     #                 print "Binning variable {} with binEdges {}\r".format(var_name,binEdges),
                    binData[var_name] = pd.cut(binData[var_name],binEdges,
                        labels=label2)   ####  labels=['bottom','top']
                    feature_map[var_name] = str(lowQVal)
            else:
                raise NameError("Unexpected bin behavior for {} with binEdges: {}".format(var_name,binEdges))

        elif len(set(binEdges)) == 2:
            binEdges = [minVal, (minVal + maxVal) / 2 , maxVal]
            #print(binEdges)
            #print(binData[var_name].isna().sum())
            binData[var_name] = pd.cut(binData[var_name], binEdges, labels = label2)
            feature_map[var_name] = str((minVal + maxVal) / 2 )
            #print(binData[var_name].isna().sum())
        elif len(set(binEdges)) == 1:
            raise ValueError(' unique value, wrong')
        else:
            raise NameError('Error in binning data for variable {}. More than 4 unique bin edges: {}'.format(var_name,binEdges))    
    
    out = data[colsDontBin].merge(binData,left_index = True,right_index = True,how = 'left')
    
    return out, feature_map






