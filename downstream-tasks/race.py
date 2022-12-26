import sys

## First append the parent directory.
sys.path.append('/home/ag82/composed-attention/')

from vanilla_transformer import Transformer
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import argparse
import time
import pdb ## For debugging only, TODO remove.

from pre_train_wiki_loader import en_tokenizer
from stats import Stats 
from constants import Constants
import tensorflow_models as tfm
from tokenization_proc import pad, add_start_end

## Basic command line arguments. ##
parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA', 'CompMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')
parser.add_argument('--step_count', dest='num_steps', type=int, default=500000, help='the number of steps as input to pre-training.')
parser.add_argument('--rank', dest='rank', type=int, default=1, help='The rank of the process, to distinguish output.')
parser.add_argument('--encoder_only', dest='enc_only', action='store_true', help='Whether we are training in encoder only mode')
parser.add_argument('--warmup', dest='warmup', default=10000, type=int, help='The number of warmup steps required during pre-training.')

args = parser.parse_args()

## Define global vars here. ##

BUFFER_SIZE = 20000

MAX_TOKENS = args.sequence_length

BATCH_SIZE = args.batch_size

### Data preparation 

curr_dir = os.getcwd() + "/"

## Load the IMDB dataset. ##
train_data, val_data, test_data = tfds.load(name="race", split=('train[:60%]', 'test[60%:]', 'test'))

## RACE is fundamentally different in that we have to build our own custom data loader. ##
class RaceLoader:
    def __init__(self, train_data, val_data, batch_size):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        
        self.train_iter = iter(train_data)
        self.val_iter = iter(val_data)
        
        self.has_train_stopped = False
        self.has_val_stopped = False
        
    def has_more_train_data(self):
        return not self.has_train_stopped
        
    def reset_train_data(self):
        self.has_train_stopped = False
        self.train_iter = iter(train_data)
        
    def gen_next_train_batch(self):
        assert self.has_train_stopped == False, "No more training data is left! Please call the reset function"
        ## Initialize empty tensors.
        enc_tensor, dec_tensor, answer_tensor = None, None, None
        try:
            count = 0
            while count < self.batch_size:
                if count == 0:
                    enc_tensor, dec_tensor, answer_tensor = self.prepare_single_inp(next(self.train_iter))
                else:
                    enc_inp, dec_inp, ans_inp = self.prepare_single_inp(next(self.train_iter))
                    enc_tensor = tf.concat([enc_tensor, enc_inp], axis=0)
                    dec_tensor = tf.concat([dec_tensor, dec_inp], axis=0)
                    answer_tensor = tf.concat([answer_tensor, ans_inp], axis=0)
                count += 1                
        except Exception as e:
            ## Over here we have finished iterating over the dataset.
            
            ## TODO, assert that e is indeed a StopIteration error.
            
            ## We set the boolean flag of whether has_train_stopped to True.
            self.has_train_stopped = True
            return enc_tensor, dec_tensor, answer_tensor
        
        return enc_tensor, dec_tensor, answer_tensor

        
    def prepare_single_inp(self, inp_dict): ## inp_dict mapping: {answers, article, example_id, options, questions}
        article = inp_dict['article']
        options = inp_dict['options']
        questions = inp_dict['questions']
        answers = inp_dict['answers']
        article_np = article.numpy()
        questions_np = questions.numpy()[:, :1][0] ## Grab the first question.
        answers_np = answers.numpy()[:, :1][0] ## Grab the first answer.
        options_np = options.numpy()[:, :1][0][0] ## Grab the list of the first options.
        enc_str = article_np[0] + b' [CLASS] ' + questions_np[0] + b' [CLASS] '
        for idx, opt in enumerate(options_np):
            if idx == len(options_np) - 1: ## This is the last concat we are doing.
                enc_str += opt
            else:
                enc_str += opt + b' [SEP] '
        ## We return a tensor of: concat(article, [SEP], question)
        ## alognside the answer
        dec_str = answers_np[0]
        answer_str = answers_np[0]
        return tf.convert_to_tensor([[enc_str]]), tf.convert_to_tensor([[dec_str]]), tf.convert_to_tensor([[answer_str]])

## We instantiate and create our data Loader. ##
train_data = train_data.batch(1)
val_data = val_data.batch(1)
raceLoader = RaceLoader(train_data, val_data, args.batch_size)

## Now, we have to create our downstream model. ##
class DownstreamModel(tf.keras.Model):

    def __init__(self, transformer):
        super(DownstreamModel, self).__init__()

        self.transformer = transformer

        ## Then, we create additional FFNs. ##
        self.layer_one = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu) ## Another hyperparameter that we must tune. 
        self.layer_two = tf.keras.layers.Dense(4, activation=tf.keras.activations.sigmoid) ## The last layer should have four options.

    def call(self, inp):
        out, _ = self.transformer(inp)
        out = self.layer_one(out)
        out = self.layer_two(out)

        return out

## We define our transformer here. ##
num_layers = args.layers
d_model = 512
dff = 2048
num_attention_heads = 8
dropout_rate = 0.1
rank = args.rank

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_attention_heads=num_attention_heads,
    dff=dff,
    input_vocab_size=Constants.wiki_vocab_size,
    target_vocab_size=Constants.wiki_vocab_size,
    dropout_rate=dropout_rate,
    downsampling_value=args.downsampling_k if args.attention_type == 'LinMHA' or args.attention_type == 'CompMHA' else 32, # Just default to 32 otherwise, doesn't matter since it won't be used.
    attention_type=args.attention_type,
    sequence_length=args.sequence_length,
    encoder_only=args.enc_only)

## Then, we create the learning rate schedule. ##
initial_learning_rate = 0.0001
num_train_steps = args.num_steps
warmup_steps = args.warmup
linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
optimizer = tf.keras.optimizers.Adam(warmup_schedule, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-9)

## We create our loss function. ##

## TODO, check for correctness. ##
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

## TODO, come back to this later. ## TODO, sanity check this.
def accuracy_function(real, pred):
  ## We delete whatever corresponds to the [END] token over here. 
  accuracies = tf.math.equal(tf.cast(real, dtype=tf.int64), tf.cast(tf.round(pred), dtype=tf.int64))
  accuracies = tf.cast(accuracies, dtype=tf.float32)
  return tf.reduce_sum(accuracies) / tf.reduce_sum(tf.ones(shape=real.shape, dtype=tf.float32)) ## Divide by batch * seq to get accuracy over everything.
  
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(initial_learning_rate) + '/' + str(args.warmup)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
else:
    ## Otherwise we raise an exception. We should ALWAYS load from a checkpoint over here. ##
    raise Exception

train_step_signature = [
    (
         tf.TensorSpec(shape=(None, None), dtype=tf.int64),
         tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

## Now, we can instantiate our downstream model. ##
downstream_model = DownstreamModel(transformer)

## TODO, this will need to be changed. ##
def pad_data(enc_inp, dec_inp, answer):
  global MAX_TOKENS, en_tokenizer
  """
  Pads the vector inputs (of size (BATCH_SIZE, SEQUENCE LENGTH)) to ensure each
  sequence length is standardized to MAX_TOKENS.
  """
  enc_tok = en_tokenizer.tokenize(enc_inp)
  dec_tok = en_tokenizer.tokenize(dec_inp)

  enc_inp = enc_tok.merge_dims(-2, -1).merge_dims(-2, -1).to_tensor()
  dec_inp = dec_tok.merge_dims(-2, -1).merge_dims(-2, -1).to_tensor()

  enc_inp = enc_inp[:, :MAX_TOKENS]
  enc_inp = pad(enc_inp, MAX_TOKENS)

  ## add the start and end tokens. ##
  enc_inp = enc_inp[:, :-2]
  enc_inp = add_start_end(enc_inp)
  
  dec_inp = add_start_end(dec_inp)

  answer = dec_inp[:, 1:] ## Remove the first token from the answer.
  dec_inp = dec_inp[:, :-1] ## remove the last token from the decoder output.

  return enc_inp, dec_inp, answer ## Also remove the last token from tar_real.

def train_step(enc_inp, dec_inp, answer):

  enc_inp, dec_inp, answer = pad_data(enc_inp, dec_inp, answer)

  with tf.GradientTape() as tape:
    predictions, _ = downstream_model([enc_inp, dec_inp],
                                      training = True)
    ## we have to recast the predictions. 
    loss = loss_object(answer, predictions) 
    accuracy = accuracy_function(answer, predictions)

  gradients = tape.gradient(loss, downstream_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, downstream_model.trainable_variables))

  train_loss.update_state(loss)
  train_accuracy(accuracy)

EPOCHS = 30
total_steps_required = args.num_steps

steps_elapsed = 0

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()
  batch = 0
  if steps_elapsed > total_steps_required:
    break

  train_loss.reset_states()
  train_accuracy.reset_states()

  ## We set a debugging statement over here. ##
  pdb.set_trace()
  while raceLoader.has_more_train_data():
    if steps_elapsed > total_steps_required:
      break
    enc_inp, dec_inp, answer = raceLoader.gen_next_train_batch()
    train_step(enc_inp, dec_inp, answer)
    if (steps_elapsed % 1000 == 0):
      # We print end-to-end time here just in case.
      print(f'----------- End-to-End: {time.time() - train_start} -----------')

    print(f'Steps {steps_elapsed} Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

    with open(f'./train_data_{args.attention_type}_{args.rank}.txt', 'a+') as f:
      f.write(f'{steps_elapsed} {train_loss.result():.4f} {train_accuracy.result():.4f}\n')

    with open(f'./train_stats_{args.attention_type}_{args.rank}.txt', 'a+') as f:
        f.write(f'{steps_elapsed} MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

    steps_elapsed += 1
    batch += 1

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)
  
  ## Lastly, we reset the training data. ##
  raceLoader.reset_train_data()

train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)

