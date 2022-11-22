## This is a python file that we will use to debug models and input data pipelines. ##


## it looks like the data pipeline is a little buggy, let us debug it. ##

from pre_train_wiki_loader import get_dataset, make_batches



train_ds = get_dataset()

BUFFER_SIZE = 20000
BATCH_SIZE = 32
train_batches = make_batches(train_ds, BUFFER_SIZE, BATCH_SIZE)

## To sanity check how many iterations it takes to finish one epoch. ##
for batch, inp in enumerate(train_batches):
    print(batch)

print('finished!')