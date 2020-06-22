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


tf.config.experimental_run_functions_eagerly(True)

'''
python Transformer_keras.py --load_from_npy --batch_size 256 --epoch 5 --num_transformer 1 --head_attention 1 --num_lstm 1 --examples 100000
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
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=int,
                    help='选择GPU',
                    default=0)
args = parser.parse_args()
if args.CUDA_VISIBLE_DEVICES == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.CUDA_VISIBLE_DEVICES == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
NUM_creative_id = 3412772
NUM_ad_id = 3027360
NUM_product_id = 39057
NUM_advertiser_id = 57870
NUM_industry = 332
NUM_product_category = 18

vocab_size = 5000
max_seq_len = 150

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
                   trainable=args.not_train_embedding,#
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=64,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,#
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=16,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,#ttt
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=8,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,###ttt
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
                   trainable=args.not_train_embedding,#
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=64,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,#
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=16,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=150,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=8,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,
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
            'age': losses.CategoricalCrossentropy(from_logits=False)#
        },
        # loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    return model


def get_train_val():

    # 提取词向量文件
    def get_embedding(feature_name, tokenizer):
        path = f'C:/Users/yrqun/Desktop/TMP/w2c_data/150/{feature_name}.kv'
        feature_name_dict = {'creative_id':256,'ad_id':256,'advertiser_id':64,'product_id':32,'product_category':8,'industry':16}
        wv = KeyedVectors.load(path, mmap='r')
        feature_tokens = list(wv.vocab.keys())
        embedding_dim = feature_name_dict[feature_name]
        embedding_matrix = np.random.randn(
            len(feature_tokens)+1, embedding_dim)
        for feature in feature_tokens:
            embedding_vector = wv[feature]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([feature])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

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

        sequences = tokenizer.texts_to_sequences(feature_seq[:900000//1])
        X_train = pad_sequences(
            sequences, maxlen=len_feature, padding='post')
        return X_train, tokenizer

    # 构造输出的训练标签
    # 获得age、gender标签
    DATA = {}

    user_train = pd.read_csv(
        f'C:/Users/yrqun/Desktop/TMP/data_raw/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
    Y_gender = user_train['gender'].values
    Y_age = user_train['age'].values
    Y_gender = Y_gender - 1
    Y_age = Y_age - 1
    Y_age = to_categorical(Y_age)
    Y_gender = to_categorical(Y_gender)

    num_examples = Y_age.shape[0]
    train_examples = int(num_examples * (1-0.001))

    DATA['Y_gender_train'] = Y_gender[:train_examples]
    DATA['Y_gender_val'] = Y_gender[train_examples:]
    DATA['Y_age_train'] = Y_age[:train_examples]
    DATA['Y_age_val'] = Y_age[train_examples:]

    # 第一个输入
    print('获取 creative_id 特征')
    X1_train, tokenizer = get_train(
        'creative_id', NUM_creative_id+1, LEN_creative_id)  # +1为了UNK的creative_id
    creative_id_emb = get_embedding('creative_id', tokenizer)

    DATA['X1_train'] = X1_train[:train_examples]
    DATA['X1_val'] = X1_train[train_examples:]
    DATA['creative_id_emb'] = creative_id_emb

    # 第二个输入
    print('获取 ad_id 特征')
    X2_train, tokenizer = get_train(
        'ad_id', NUM_ad_id+1, LEN_ad_id)
    ad_id_emb = get_embedding('ad_id', tokenizer)

    DATA['X2_train'] = X2_train[:train_examples]
    DATA['X2_val'] = X2_train[train_examples:]
    DATA['ad_id_emb'] = ad_id_emb

    # 第三个输入
    print('获取 product_id 特征')
    X3_train, tokenizer = get_train(
        'product_id', NUM_product_id+1, LEN_product_id)
    product_id_emb = get_embedding('product_id', tokenizer)

    DATA['X3_train'] = X3_train[:train_examples]
    DATA['X3_val'] = X3_train[train_examples:]
    DATA['product_id_emb'] = product_id_emb

    # 第四个输入
    print('获取 advertiser_id 特征')
    X4_train, tokenizer = get_train(
        'advertiser_id', NUM_advertiser_id+1, LEN_advertiser_id)
    advertiser_id_emb = get_embedding('advertiser_id', tokenizer)

    DATA['X4_train'] = X4_train[:train_examples]
    DATA['X4_val'] = X4_train[train_examples:]
    DATA['advertiser_id_emb'] = advertiser_id_emb

    # 第五个输入
    print('获取 industry 特征')
    X5_train, tokenizer = get_train(
        'industry', NUM_industry+1, LEN_industry)
    industry_emb = get_embedding('industry', tokenizer)

    DATA['X5_train'] = X5_train[:train_examples]
    DATA['X5_val'] = X5_train[train_examples:]
    DATA['industry_emb'] = industry_emb

    # 第六个输入
    print('获取 product_category 特征')
    X6_train, tokenizer = get_train(
        'product_category', NUM_product_category+1, LEN_product_category)
    product_category_emb = get_embedding('product_category', tokenizer)

    DATA['X6_train'] = X6_train[:train_examples]
    DATA['X6_val'] = X6_train[train_examples:]
    DATA['product_category_emb'] = product_category_emb

    return DATA


# %%
if not args.load_from_npy:
    print('从csv文件提取训练数据到array格式，大概十几分钟时间')
    DATA = get_train_val()

    # 训练数据保存为npy文件
    dirs = 'tmp/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    def save_npy(datas, name):
        for i, data in enumerate(datas):
            np.save(f'tmp/{name}_{i}.npy', data)
            print(f'saving tmp/{name}_{i}.npy')

    inputs = [
        DATA['X1_train'], DATA['X1_val'],
        DATA['X2_train'], DATA['X2_val'],
        DATA['X3_train'], DATA['X3_val'],
        DATA['X4_train'], DATA['X4_val'],
        DATA['X5_train'], DATA['X5_val'],
        DATA['X6_train'], DATA['X6_val'],
    ]
    outputs_gender = [DATA['Y_gender_train'], DATA['Y_gender_val']]
    outputs_age = [DATA['Y_age_train'], DATA['Y_age_val']]
    embeddings = [
        DATA['creative_id_emb'],
        DATA['ad_id_emb'],
        DATA['product_id_emb'],
        DATA['advertiser_id_emb'],
        DATA['industry_emb'],
        DATA['product_category_emb'],
    ]
    save_npy(inputs, 'inputs')
    save_npy(outputs_gender, 'gender')
    save_npy(outputs_age, 'age')
    save_npy(embeddings, 'embeddings')
else:
    DATA = {}
    DATA['X1_train'] = np.load('tmp/inputs_0.npy', allow_pickle=True)
    DATA['X1_val'] = np.load('tmp/inputs_1.npy', allow_pickle=True)
    DATA['X2_train'] = np.load('tmp/inputs_2.npy', allow_pickle=True)
    DATA['X2_val'] = np.load('tmp/inputs_3.npy', allow_pickle=True)
    DATA['X3_train'] = np.load('tmp/inputs_4.npy', allow_pickle=True)
    DATA['X3_val'] = np.load('tmp/inputs_5.npy', allow_pickle=True)
    DATA['X4_train'] = np.load('tmp/inputs_6.npy', allow_pickle=True)
    DATA['X4_val'] = np.load('tmp/inputs_7.npy', allow_pickle=True)
    DATA['X5_train'] = np.load('tmp/inputs_8.npy', allow_pickle=True)
    DATA['X5_val'] = np.load('tmp/inputs_9.npy', allow_pickle=True)
    DATA['X6_train'] = np.load('tmp/inputs_10.npy', allow_pickle=True)
    DATA['X6_val'] = np.load('tmp/inputs_11.npy', allow_pickle=True)
    DATA['Y_gender_train'] = np.load('tmp/gender_0.npy', allow_pickle=True)
    DATA['Y_gender_val'] = np.load('tmp/gender_1.npy', allow_pickle=True)
    DATA['Y_age_train'] = np.load('tmp/age_0.npy', allow_pickle=True)
    DATA['Y_age_val'] = np.load('tmp/age_1.npy', allow_pickle=True)
    DATA['creative_id_emb'] = np.load(
        'tmp/embeddings_0.npy', allow_pickle=True)
    DATA['ad_id_emb'] = np.load(
        'tmp/embeddings_1.npy', allow_pickle=True)
    DATA['product_id_emb'] = np.load(
        'tmp/embeddings_2.npy', allow_pickle=True)
    DATA['advertiser_id_emb'] = np.load(
        'tmp/embeddings_3.npy', allow_pickle=True)
    DATA['industry_emb'] = np.load(
        'tmp/embeddings_4.npy', allow_pickle=True)
    DATA['product_category_emb'] = np.load(
        'tmp/embeddings_5.npy', allow_pickle=True)


# %%

# # %%
if args.gender:
    try:
        checkpoint = ModelCheckpoint("tmp/gender_epoch_{epoch:02d}.hdf5", save_weights_only=True, monitor='val_loss', verbose=1,
                                     save_best_only=False, mode='auto', period=2)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.00001,
            patience=3,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                                  factor=0.5,
                                                                  patience=1,
                                                                  min_lr=0.0000001)
        model = get_gender_model(DATA)
        model.summary()

        train_examples = args.train_examples
        val_examples = args.val_examples
        model.fit(
            {
                'creative_id': DATA['X1_train'][:train_examples],
                'ad_id': DATA['X2_train'][:train_examples],
                'product_id': DATA['X3_train'][:train_examples],
                'advertiser_id': DATA['X4_train'][:train_examples],
                'industry': DATA['X5_train'][:train_examples],
                'product_category': DATA['X6_train'][:train_examples]
            },
            {
                'gender': DATA['Y_gender_train'][:train_examples],
                # 'age': DATA['Y_age_train'][:train_examples],
            },
            validation_data=(
                {
                    'creative_id': DATA['X1_val'][:val_examples],
                    'ad_id': DATA['X2_val'][:val_examples],
                    'product_id': DATA['X3_val'][:val_examples],
                    'advertiser_id': DATA['X4_val'][:val_examples],
                    'industry': DATA['X5_val'][:val_examples],
                    'product_category': DATA['X6_val'][:val_examples]
                },
                {
                    'gender': DATA['Y_gender_val'][:val_examples],
                    # 'age': DATA['Y_age_val'][:val_examples],
                },
            ),
            epochs=args.epoch,
            batch_size=args.batch_size,
            # callbacks=[checkpoint, reduce_lr_callback],
            callbacks=[checkpoint, earlystop_callback, reduce_lr_callback],
        )
    except Exception as e:
        # e = str(e)
        print(e)
elif args.age:
    try:
        checkpoint = ModelCheckpoint("tmp/age_epoch_{epoch:02d}.hdf5", save_weights_only=True, monitor='val_loss', verbose=1,
                                     save_best_only=False, mode='auto', period=2)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.00001,
            patience=3,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                                  factor=0.5,
                                                                  patience=1,
                                                                  min_lr=0.0000001)
        model = get_age_model(DATA)
        model.summary()

        train_examples = args.train_examples
        val_examples = args.val_examples
        model.fit(
            {
                'creative_id': DATA['X1_train'][:train_examples],
                'ad_id': DATA['X2_train'][:train_examples],
                'product_id': DATA['X3_train'][:train_examples],
                'advertiser_id': DATA['X4_train'][:train_examples],
                'industry': DATA['X5_train'][:train_examples],
                'product_category': DATA['X6_train'][:train_examples]
            },
            {
                # 'gender': DATA['Y_gender_train'][:train_examples],
                'age': DATA['Y_age_train'][:train_examples],
            },
            validation_data=(
                {
                    'creative_id': DATA['X1_val'][:val_examples],
                    'ad_id': DATA['X2_val'][:val_examples],
                    'product_id': DATA['X3_val'][:val_examples],
                    'advertiser_id': DATA['X4_val'][:val_examples],
                    'industry': DATA['X5_val'][:val_examples],
                    'product_category': DATA['X6_val'][:val_examples]
                },
                {
                    # 'gender': DATA['Y_gender_val'][:val_examples],
                    'age': DATA['Y_age_val'][:val_examples],
                },
            ),
            epochs=args.epoch,
            batch_size=args.batch_size,
            # callbacks=[checkpoint, reduce_lr_callback],
            callbacks=[checkpoint, earlystop_callback, reduce_lr_callback],
        )
    except Exception as e:
        print(e)
# %%
# model.load_weights('tmp/gender_epoch_01.hdf5')


# # %%
# if debug:
#     sequences = tokenizer.texts_to_sequences(
#         creative_id_seq[900000:])
# else:
#     sequences = tokenizer.texts_to_sequences(
#         creative_id_seq[900000:])

# X_test = pad_sequences(sequences, maxlen=LEN_creative_id)
# # %%
# y_pred = model.predict(X_test, batch_size=4096)

# y_pred = np.where(y_pred > 0.5, 1, 0)
# y_pred = y_pred.flatten()

# # %%
# y_pred = y_pred+1
# # %%
# res = pd.DataFrame({'predicted_gender': y_pred})
# res.to_csv(
#     'data/ans/lstm_gender.csv', header=True, columns=['predicted_gender'], index=False)


# # %%

# %%