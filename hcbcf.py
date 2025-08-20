import pandas as pd

import helpers
fail_value = .5

def getHCBCFiSimContent(item1, item2):
    parts1 = item1.split('.')
    parts2 = item2.split('.')
    if parts1[0] == parts2[0]:
        if len(parts1) > 1 and len(parts2) > 1 and parts1[1] == parts2[1]:
            return 1
        else:
            return 0.5
    else:
        return 0

def getHCBCFCFPredict(df, user, item, neighborN=None) :
    # fail_value = pd.to_numeric(df.iloc[1:, 1:].stack(), errors='coerce').mean()
    if neighborN is None :
        neighborN = 4
    ratedItemsByUser = helpers.getUsersRatedItems(df, user)
    similarities = []
    for item2 in ratedItemsByUser:
        iSimCF = helpers.getHCBCFiSimCF(df, item, item2)
        similarities.append((item2, iSimCF))
    similarities.sort(key=lambda x: x[1], reverse=True)
    topNItems = similarities[:neighborN]
    numerator = denominator = 0
    for item2, iSimCF in topNItems:
        numerator += helpers.getRating(df, user, item2) * iSimCF
        denominator += abs(iSimCF)
    
    # if denominator == 0:
    #     print('HCBCF failed')
    # return numerator/denominator if denominator != 0 else 0
    return numerator/denominator if denominator != 0 else fail_value

def getHCBCFContentPredict(df, user, item, neighborN=None):
    # fail_value = pd.to_numeric(df.iloc[1:, 1:].stack(), errors='coerce').mean()
    if neighborN is None :
        neighborN = 4
    ratedItemsByUser = helpers.getUsersRatedItems(df, user)
    similarities = []
    for item2 in ratedItemsByUser:
        iSimContent = getHCBCFiSimContent(item, item2)
        similarities.append((item2, iSimContent))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_similarities = similarities[:neighborN]
    numerator = denominator = 0
    for item2, iSimContent in top_n_similarities:
        numerator += helpers.getRating(df, user, item2) * iSimContent
        denominator += abs(iSimContent)
    
    # if denominator == 0:
    #     print('HCBCF failed')
    # return numerator / denominator if denominator != 0 else 0
    return numerator / denominator if denominator != 0 else fail_value

def getHCBCFHybridPredict(df, user, item, lambda_param=None, nn=None) :
    if lambda_param is None :
        lambda_param = 0.5
    return lambda_param * getHCBCFCFPredict(df, user, item, neighborN=nn) + (1 - lambda_param) * getHCBCFContentPredict(df, user, item, neighborN=nn)