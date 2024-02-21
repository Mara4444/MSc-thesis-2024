
import pandas as pd
import math

################################################
####            language similarity         ####
################################################

def get_wals_data(walsdata,lang):
    """
    Get the wals dataframe row of a specific language.
    
    Parameters:
    lang: wals_code of the language of interest.
    
    Returns:
    pd.series row of the wals data for the language.
    """
    return walsdata.loc[walsdata['wals_code'] == lang] # .values.flatten().tolist() get list of walsdata row for a requested language
   
def calc_distance(walsdata,lang1,lang2):
    """
    Calculate the Haversine distance between the geographical locations of two 
    languages given the wals_codes.
    
    Parameters:
    lang1, lang2: wals_code of two languages to be compared.
    
    Returns:
    Rounded distance between the geographical location of the languages in kilometers.
    """

    # get df for both languages
    wals_lang1 = get_wals_data(walsdata,lang1)
    wals_lang2 = get_wals_data(walsdata,lang2)
    
    lat1 = wals_lang1['latitude']
    lon1 = wals_lang1['longitude']
    lat2 = wals_lang2['latitude']
    lon2 = wals_lang2['longitude']

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of the Earth (mean value in kilometers)
    radius_earth = 6371

    # Calculate the distance
    distance = radius_earth * c

    return round(distance)

def get_distance_df(walsdata,langlist):
    """
    Calculate the Haversine distance between the geographical locations of a 
    list of languages with the wals_codes and creates a df with the pairwise
    distances.
    
    Parameters:
    langlist: list with wals_codes of languages to be compared.
    
    Returns:
    Rounded distance between the geographical location of the languages in kilometers.
    """
    df = pd.DataFrame(index=langlist, columns=langlist)

    for lang1 in langlist:
        for lang2 in langlist:
            # print(lang1,lang2,calc_distance(lang1,lang2))
            df.loc[lang1,lang2] = calc_distance(walsdata,lang1,lang2)

    return df

def get_sim_df(walsdata,langlist,featurelist):
    """
    Create a df of similarity between a list of languages based on feature set. The similarity 
    is a sum of binary value (1: feature values are equal for the language pair, 0: else) divided 
    by the number of features that are filled in for both languages.
    
    Parameters:
    langlist: list with wals_codes of languages to be compared.
    
    Returns:
    Df with shared overlap value between each language.
    """
    df = pd.DataFrame(index=langlist, columns=langlist)
    df = df.fillna(value=0.0)

    for lang1 in langlist:

        for lang2 in langlist:

            features_count = 0

            # get df for both languages
            wals_lang1 = get_wals_data(walsdata,lang1)
            wals_lang2 = get_wals_data(walsdata,lang2)
            
            for feature in featurelist:
                # count1 = pd.notna(wals_lang1.at[wals_lang1.index[0], feature]).sum()
                if pd.notna(wals_lang1.at[wals_lang1.index[0], feature]) and pd.notna(wals_lang2.at[wals_lang2.index[0], feature]): # only check equality for language features that are not nan
                   
                    features_count += 1

                    value = 1 if wals_lang1.at[wals_lang1.index[0],feature] == wals_lang2.at[wals_lang2.index[0],feature] else 0 # set 1 for equal value, 0 for unequal

                    df.loc[lang1,lang2] += value
            
            # divide by nr of filled features by both languages
            if features_count > 0:
                df.loc[lang1, lang2] = round(df.loc[lang1, lang2] / features_count,2)
    
    return df 