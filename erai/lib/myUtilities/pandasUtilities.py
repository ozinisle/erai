def oneHotEncode_and_bind(original_dataframe, feature_to_encode):
    import pandas as pd 

    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

def oneHotEncode_ExternalData(original_dataframe, external_feature_to_encode,col_name_prefix=''):
    import pandas as pd 
    
    dummies = pd.get_dummies(external_feature_to_encode).add_prefix(col_name_prefix)
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)