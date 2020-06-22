# %%
# 生成词嵌入文件
from layers import Add, LayerNormalization
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import PositionEncoding
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, Concatenate, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec, KeyedVectors
from mymail import mail


tf.config.experimental_run_functions_eagerly(True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
python Transformer_keras_6_input_predict.py --load_from_npy --age --gender
'''

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--load_from_npy', action='store_true',
                    help='从npy文件加载数据',
                    default=False)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',

                    default=True)
parser.add_argument('--gender', action='store_true',
                    help='gender model',
                    default=False)
parser.add_argument('--age', action='store_true',
                    help='age model',
                    default=False)

parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--predict', action='store_true',
                    help='从npy文件加载数据',
                    default=False)

parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--head_attention', type=int,
                    help='transformer head个数',
                    default=1)

parser.add_argument('--train_examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)
parser.add_argument('--val_examples', type=int,
                    help='验证集数据，调试时候可以设置1000',
                    default=90000)
args = parser.parse_args()
# %%
NUM_creative_id = 3412772
NUM_ad_id = 3027360
NUM_product_id = 39057
NUM_advertiser_id = 57870
NUM_industry = 332
NUM_product_category = 18

LEN_creative_id = 150
LEN_ad_id = 150
LEN_product_id = 150
LEN_advertiser_id = 150
LEN_industry = 150
LEN_product_category = 150

# %%


def get_gender_model(DATA):

    feed_forward_size = 2048
    max_seq_len = 150
    model_dim = 256+256+64+32+8+16

    input_creative_id = Input(shape=(max_seq_len,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
                   output_dim=256,
                   weights=[DATA['creative_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_creative_id)
    # encodings = PositionEncoding(model_dim)(x1)
    # encodings = Add()([embeddings, encodings])

    input_ad_id = Input(shape=(max_seq_len,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id+1,
                   output_dim=256,
                   weights=[DATA['ad_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_ad_id)

    input_product_id = Input(shape=(max_seq_len,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=32,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=64,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=16,
                   weights=[DATA['industry_emb']],
                   trainable=True,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=8,
                   weights=[DATA['product_category_emb']],
                   trainable=True,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_category)

    # (bs, 100, 128*2)
    encodings = layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6])
    # (bs, 100)
    masks = tf.equal(input_creative_id, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(8, 79)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim, feed_forward_size)
    ff_out = ff(attention_out)
    # Add & Norm
    # ff_out (bs, 100, 128)，但是attention_out是(bs,100,256)
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_gender = Dense(2, activation='softmax', name='gender')(encodings)
    # output_age = Dense(10, activation='softmax', name='age')(encodings)

    model = Model(
        inputs=[input_creative_id,
                input_ad_id,
                input_product_id,
                input_advertiser_id,
                input_industry,
                input_product_category],
        outputs=[output_gender]
    )

    model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={
            'gender': losses.CategoricalCrossentropy(from_logits=False),
            # 'age': losses.CategoricalCrossentropy(from_logits=False)
        },
        # loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    return model


def get_age_model(DATA):

    feed_forward_size = 2048
    max_seq_len = 150
    model_dim = 256+256+64+32+8+16

    input_creative_id = Input(shape=(max_seq_len,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
                   output_dim=256,
                   weights=[DATA['creative_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_creative_id)
    # encodings = PositionEncoding(model_dim)(x1)
    # encodings = Add()([embeddings, encodings])

    input_ad_id = Input(shape=(max_seq_len,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id+1,
                   output_dim=256,
                   weights=[DATA['ad_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_ad_id)

    input_product_id = Input(shape=(max_seq_len,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=32,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=64,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=16,
                   weights=[DATA['industry_emb']],
                   trainable=True,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=8,
                   weights=[DATA['product_category_emb']],
                   trainable=True,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_category)

    # (bs, 100, 128*2)
    encodings = layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6])
    # (bs, 100)
    masks = tf.equal(input_creative_id, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(8, 79)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim, feed_forward_size)
    ff_out = ff(attention_out)
    # Add & Norm
    # ff_out (bs, 100, 128)，但是attention_out是(bs,100,256)
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    # output_gender = Dense(2, activation='softmax', name='gender')(encodings)
    output_age = Dense(10, activation='softmax', name='age')(encodings)

    model = Model(
        inputs=[input_creative_id,
                input_ad_id,
                input_product_id,
                input_advertiser_id,
                input_industry,
                input_product_category],
        outputs=[output_age]
    )

    model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={
            # 'gender': losses.CategoricalCrossentropy(from_logits=False),
            'age': losses.CategoricalCrossentropy(from_logits=False)
        },
        # loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    return model


def get_train_val():

    # 从序列文件提取array格式数据
    def get_train(feature_name, vocab_size, len_feature):
        f = open(f'C:/Users/yrqun/Desktop/TMP/w2c_data/150/{feature_name}.txt')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(f)
        f.close()

        feature_seq = []
        with open(f'C:/Users/yrqun/Desktop/TMP/w2c_data/150/{feature_name}.txt') as f:
            for text in f:
                feature_seq.append(text.strip())

        sequences = tokenizer.texts_to_sequences(feature_seq[900000:])
        X_test = pad_sequences(
            sequences, maxlen=len_feature, padding='post')
        return tokenizer, X_test

    DATA = {}
    # 获取test数据

    # 第一个输入
    print('获取 creative_id 特征')
    tokenizer, X1_test = get_train(
        'creative_id', NUM_creative_id+1, LEN_creative_id)  # +1为了UNK的creative_id

    DATA['X1_test'] = X1_test

    # 第二个输入
    print('获取 ad_id 特征')
    tokenizer, X2_test = get_train(
        'ad_id', NUM_ad_id+1, LEN_ad_id)

    DATA['X2_test'] = X2_test

    # 第三个输入
    print('获取 product_id 特征')
    tokenizer, X3_test = get_train(
        'product_id', NUM_product_id+1, LEN_product_id)

    DATA['X3_test'] = X3_test

    # 第四个输入
    print('获取 advertiser_id 特征')
    tokenizer, X4_test = get_train(
        'advertiser_id', NUM_advertiser_id+1, LEN_advertiser_id)

    DATA['X4_test'] = X4_test

    # 第五个输入
    print('获取 industry 特征')
    tokenizer, X5_test = get_train(
        'industry', NUM_industry+1, LEN_industry)

    DATA['X5_test'] = X5_test

    # 第六个输入
    print('获取 product_category 特征')
    tokenizer, X6_test = get_train(
        'product_category', NUM_product_category+1, LEN_product_category)

    DATA['X6_test'] = X6_test

    return DATA


# %%
if not args.load_from_npy:
    print('从csv文件提取训练数据到array格式，大概十几分钟时间')
    DATA = get_train_val()

    # 训练数据保存为npy文件
    dirs = 'C:/Users/yrqun/Desktop/TMP/trans/tmp'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    def save_npy(datas, name):
        for i, data in enumerate(datas):
            np.save(
                f'C:/Users/yrqun/Desktop/TMP/trans/tmp/{name}_{i}.npy', data)
            print(
                f'saving C:/Users/yrqun/Desktop/TMP/trans/tmp/{name}_{i}.npy')

    test = [DATA['X1_test'],
            DATA['X2_test'],
            DATA['X3_test'],
            DATA['X4_test'],
            DATA['X5_test'],
            DATA['X6_test'], ]
    save_npy(test, 'test')
else:
    DATA = {}

    DATA['X_test1'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_0.npy', allow_pickle=True)
    DATA['X_test2'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_1.npy', allow_pickle=True)
    DATA['X_test3'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_2.npy', allow_pickle=True)
    DATA['X_test4'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_3.npy', allow_pickle=True)
    DATA['X_test5'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_4.npy', allow_pickle=True)
    DATA['X_test6'] = np.load(
        'C:/Users/yrqun/Desktop/TMP/trans/tmp/test_5.npy', allow_pickle=True)

DATA['creative_id_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_0.npy', allow_pickle=True)
DATA['ad_id_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_1.npy', allow_pickle=True)
DATA['product_id_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_2.npy', allow_pickle=True)
DATA['advertiser_id_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_3.npy', allow_pickle=True)
DATA['industry_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_4.npy', allow_pickle=True)
DATA['product_category_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_5.npy', allow_pickle=True)

# %%
model_gender = get_gender_model(DATA)
model_age = get_age_model(DATA)

# slc
model_gender.load_weights(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/gender_epoch_01.hdf5')
model_age.load_weights(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/age_epoch_01.hdf5')

y_pred_gender = model_gender.predict(
    {
        'creative_id': DATA['X1_test'],
        'ad_id': DATA['X2_test'],
        'product_id': DATA['X3_test'],
        'advertiser_id': DATA['X4_test'],
        'industry': DATA['X5_test'],
        'product_category': DATA['X6_test']
    },
    batch_size=1024,
)
y_pred_gender = np.argmax(y_pred_gender, axis=1)
y_pred_gender = y_pred_gender.flatten()
y_pred_gender += 1

ans_gender = pd.DataFrame({'predicted_gender': y_pred_gender})
ans_gender.to_csv(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/transformer_gender.csv', header=True, columns=['predicted_gender'], index=False)

y_pred_age = model_age.predict(
    {
        'creative_id': DATA['X1_test'],
        'ad_id': DATA['X2_test'],
        'product_id': DATA['X3_test'],
        'advertiser_id': DATA['X4_test'],
        'industry': DATA['X5_test'],
        'product_category': DATA['X6_test']
    },
    batch_size=1024,
)
y_pred_age = np.argmax(y_pred_age, axis=1)
y_pred_age = y_pred_age.flatten()
y_pred_age += 1

ans_age = pd.DataFrame({'predicted_age': y_pred_age})
ans_age.to_csv(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/transformer_age.csv', header=True, columns=['predicted_age'], index=False)

# slc
user_id_test = pd.read_csv(
    'C:/Users/yrqun/Desktop/TMP/data_raw/test/click_log.csv').sort_values(['user_id'], ascending=(True,)).user_id.unique()
ans = pd.DataFrame({'user_id': user_id_test})

gender = pd.read_csv(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/transformer_gender.csv')
age = pd.read_csv(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/transformer_age.csv')
ans['predicted_gender'] = gender.predicted_gender
ans['predicted_age'] = age.predicted_age
ans.to_csv('C:/Users/yrqun/Desktop/TMP/trans/tmp/submission.csv', header=True, index=False,
           columns=['user_id', 'predicted_age', 'predicted_gender'])