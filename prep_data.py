import os
from random import shuffle, seed

seed(12345)
all_classes = ['hy', 'zmf', 'sgt', 'smh', 'mzd', 'fwq', 'shz', 'oyx', 'yyr', 'wzm', 'gj', 'mf', 'wxz', 'csl', 'bdsr', 'htj', 'lgq', 'lqs', 'yzq', 'lx']
shuffle(all_classes)

SUBSET_TRAIN_DIR_NAME_LEN = 8
SUBSET_TEST_DIR_NAME_LEN = 7
os.chdir('data')
for d in os.listdir():
    dir_len = len(d)
    if dir_len == SUBSET_TRAIN_DIR_NAME_LEN or dir_len == SUBSET_TEST_DIR_NAME_LEN:
        num_keep = int(d[dir_len-2:])
        to_delete = all_classes[num_keep:]
        print(os.listdir())
        for delete in to_delete:
            print(delete)
            os.unlink(f'{d}/{delete}')