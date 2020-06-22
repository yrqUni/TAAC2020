import pandas as pd
import numpy as np

DATA = {}
DATA['ad_id_emb'] = np.load(
    'C:/Users/yrqun/Desktop/TMP/trans/tmp/embeddings_0.npy', allow_pickle=True)
arr = DATA['ad_id_emb']

result = []
for i in range(arr.shape[-1]):
    result.append([np.mean(arr[:, i]), np.std(arr[:, i])])
dfi = pd.DataFrame(result, columns=['mean', 'std'])
print(dfi.describe().T)