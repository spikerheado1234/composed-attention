import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

granularity = 500 
line_types = ['-', '-', '-', '-']

def plot_loss(*args):
    global line_types
    headings = []
    line_type = 0
    for dir, heading in args:
        # We plot the loss and accuracy curves for each 
        # directory supplied.
        loss = []
        with open(dir, "r") as f:
            for i, line in enumerate(f):
                if i % granularity == 0:
                    curr_loss, _ = line.split()
                    curr_loss = float(curr_loss)
                    loss.append(curr_loss)

        plt.plot(loss, line_types[line_type])
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper right')
    plt.title('Loss Curves')
    plt.xlabel(f'step count')
    plt.ylabel('loss')

def plot_perplexity(*args):
    global line_types
    headings = []
    line_type = 0
    for dir, heading in args:
        # We plot the loss and accuracy curves for each 
        # directory supplied.
        perplexity = []
        with open(dir, "r") as f:
            for i, line in enumerate(f):
                curr_loss, _ = line.split()
                curr_perplexity = tf.math.exp(float(curr_loss))
                if curr_perplexity < 500:
                    perplexity.append(curr_perplexity)

        plt.plot(perplexity, line_types[line_type])
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper left')
    plt.title('Perplexity Curves')
    plt.ylabel('Perplexity')

def plot_accuracy(*args):
    global line_types

    headings = []
    line_type = 0
    for dir, heading in args:
        # We plot the loss and accuracy curves for each 
        # directory supplied.
        accuracy = []
        with open(dir, "r") as f:
            for i, line in enumerate(f):
                if i % granularity == 0:
                    _, curr_accuracy = line.split()
                    curr_accuracy = float(curr_accuracy)
                    accuracy.append(curr_accuracy)

        plt.plot(accuracy, line_types[line_type])
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper left')
    plt.title('Accuracy Curves')
    plt.xlabel(f'step count (per {granularity})')
    plt.ylabel('Accuracy')

def plot_performance(*args):
    headings = []
    line_type = 0

    for dir, heading in args:
        run_time = []
        seq_length = []
        downsampling_value = []
        with open(dir, "r") as f:
            for line in f:
                line_split = line.split()
                dk_val = float(line_split[-1])
                if dk_val > 8:
                    run_time.append(float(line_split[3][:-1]))
                    #seq_length.append(float(line_split[6]))
                    downsampling_value.append(dk_val)

        a, b = np.polyfit(downsampling_value, run_time, 1)
        plt.plot(downsampling_value, a*np.array(downsampling_value)+b, line_types[line_type])
        plt.scatter(downsampling_value, run_time, marker="+")
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper left')
    plt.title('End-to-End Run-Time of 1000 iterations as $d_k$ varies')
    plt.xlabel(f'$d_k$')
    plt.ylabel('Time (s)')    

#plot_loss(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/vt_512s_8bs_train_data.txt', 'Vanilla Transformer'),
#           ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/pt_512s_8bs_train_data.txt', 'Performer-VT Semi-Composed'))
#plot_accuracy(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/vt_512s_8bs_train_data.txt', 'Vanilla Transformer'),
#           ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/pt_512s_8bs_train_data.txt', 'Performer-VT Semi-Composed'))
#plot_accuracy(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/pt_train_data.txt', 'Naive Performer'),
#           ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/ptCvt_train_data.txt', 'Performer-VT Semi-Composed'))
#plot_loss(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/lt_train_data.txt', 'LinFormer'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/pt_train_data.txt', 'PerFormer'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/ltCpt_train_data.txt', 'Lin-Perf Composed'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/vt_train_data.txt', 'Vanilla Transformer'))
#plot_accuracy(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/lt_train_data.txt', 'LinFormer'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/pt_train_data.txt', 'PerFormer'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/ltCpt_train_data.txt', 'Lin-Perf Composed'),
#              ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/vt_train_data.txt', 'Vanilla Transformer'))
#plot_new_loss(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/output_one.in', 'Vanilla Transformer'))
#plot_new_accuracy(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/output_one.in', 'Vanilla Transformer'))
plot_perplexity(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/large-model/pre-train/trial-1/LinMHA_val_data.txt', 'Linformer'),
                ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/large-model/pre-train/trial-1/compMHA_val_data.txt', 'Lin-Perf Transformer'),
                ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/data/large-model/pre-train/trial-1/MHA_val_data.txt', 'Vanilla Transformer'))
#plot_performance(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/benchmark_results_LinMHA.txt', 'Linformer'),
#                 ('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/benchmark_results_CompMHA.txt', 'Lin-Perf Transformer'))
plt.show()


