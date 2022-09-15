import tensorflow as tf
from vanilla_transformer import load_data, load_tokenizer, make_batches, positional_encoding
import matplotlib.pyplot as plt

train_examples, val_examples = load_data()

tokenizers = load_tokenizer() 

for pt_examples, en_examples in train_examples.batch(3).take(1):
  print('> Examples in Portuguese:')
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))
  print()

  print('> Examples in English:')
  for en in en_examples.numpy():
    print(en.decode('utf-8'))
print('padded-batch of token ids:')
encoded = tokenizers.en.tokenize(en_examples)
for row in encoded.to_list():
    print(row)

# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

n, d = 2048, 512

pos_encoding = positional_encoding(position=n, d_model=d)

# Check the shape.
print(pos_encoding.shape)

pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot.
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

# Plot the dimensions.
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()