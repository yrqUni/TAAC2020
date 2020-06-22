import numpy as np
import pandas as pd
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import re
from tqdm import tqdm

import sys
import time

#Data_Root
#raw
test_raw_data_root = r'C:\Users\yrqun\Desktop\TMP\data_raw\test'
train_raw_data_root = r'C:\Users\yrqun\Desktop\TMP\data_raw\train_preliminary'

#Env-CSV_Data
#train
train_ad_filepath = train_raw_data_root + r'\ad.csv'
train_click_log_filepath = train_raw_data_root + r'\click_log.csv'
train_user_filepath = train_raw_data_root + r'\user.csv'
#test
test_ad_filepath = test_raw_data_root + r'\ad.csv'
test_click_log_filepath = test_raw_data_root + r'\click_log.csv'

#data
#data_path = r'C:\Users\yrqun\Desktop\TMP\w2c_data\click_times_merge_data.csv'

#word2vec
word2vec_dict_filepath = r'C:\Users\yrqun\Desktop\TMP\w2c_data\click_times.txt'
word2vec_word2vec_model_filepath = r'C:\Users\yrqun\Desktop\TMP\w2c_data\click_times.model'
word2vec_wordvectors_kv_filepath = r'C:\Users\yrqun\Desktop\TMP\w2c_data\click_times.kv'
data_vec_filepath = r'C:\Users\yrqun\Desktop\TMP\w2c_data\click_times.csv'

def data():
    train_ad = pd.read_csv(train_ad_filepath)
    print('train_ad Read Done')
    train_click_log = pd.read_csv(train_click_log_filepath)
    print('train_click_log Read Done')
    train_user = pd.read_csv(train_user_filepath)
    print('train_user Read Done')

    test_ad = pd.read_csv(test_ad_filepath)
    print('test_ad Read Done')
    test_click_log = pd.read_csv(test_click_log_filepath)
    print('test_click_log Read Done')
    
    print('\nData Read Done\n')
    
    return train_ad, train_click_log, train_user, test_ad, test_click_log

if __name__ == '__main__':
    train_ad, train_click_log, train_user, test_ad, test_click_log = data()
    
    click_log = train_click_log.append(test_click_log)
    ad = train_ad.append(test_ad).drop_duplicates(subset=None, keep='first', inplace=False)
    data = pd.merge(click_log, ad, on='creative_id', how='left').sort_values(by=['user_id', 'time', 'click_times'], ascending=[True, True, False], axis=0).fillna(int(0)).replace('\\N',int(0)).astype(int)

###
    print(click_log.shape)
    print(ad.shape)
    print(data.shape)
    print(click_log.shape[0]==data.shape[0])
###

    data_click_times = data.groupby("user_id")['click_times'].apply(list).reset_index(name='click_times')['click_times']
    with open(word2vec_dict_filepath, 'w')as f:
        with tqdm(total=int(len(data_click_times))) as pbar:
            for i in data_click_times:
                i = [str(e) for e in i]
                line = ' '.join(i)
                f.write(line+'\n')
                pbar.update(1) 
    sentences = LineSentence(word2vec_dict_filepath)
    dimension_embedding = 128
    model = Word2Vec(sentences, size=dimension_embedding, window=10, min_count=1, workers=-1, iter=10, sg=1)
    model.save(word2vec_word2vec_model_filepath)
    model.wv.save(word2vec_wordvectors_kv_filepath)
    #data.to_csv(data_path,header=True)
    print('DONE')
    
    
    