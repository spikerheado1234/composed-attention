import matplotlib.pyplot as plt


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

    plt.legend(headings, loc='upper left')
    plt.title('Loss Curves')
    plt.xlabel(f'step count')
    plt.ylabel('loss')

def plot_new_loss(*args):
    global line_types
    headings = []
    line_type = 0
    for dir, heading in args:
        # We plot the loss and accuracy curves for each 
        # directory supplied.
        loss = []
        with open(dir, "r") as f:
            for i, line in enumerate(f):
                split_tokens = line.split()
                if (len(split_tokens) > 0 and split_tokens[0] != "Steps") or (len(split_tokens) == 0):
                    continue
                loss.append(float(split_tokens[7]))

        plt.plot(loss, line_types[line_type])
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper left')
    plt.title('Loss Curves')
    plt.xlabel(f'step count')
    plt.ylabel('loss')

def plot_new_accuracy(*args):
    global line_types

    headings = []
    line_type = 0
    for dir, heading in args:
        # We plot the loss and accuracy curves for each 
        # directory supplied.
        accuracy = []
        with open(dir, "r") as f:
            for i, line in enumerate(f):
                split_tokens = line.split()
                if (len(split_tokens) > 0 and split_tokens[0] != "Steps") or (len(split_tokens) == 0):
                    continue
                curr_accuracy = float(split_tokens[9])
                accuracy.append(curr_accuracy)

        plt.plot(accuracy, line_types[line_type])
        line_type += 1
        headings.append(heading)

    plt.legend(headings, loc='upper left')
    plt.title('Accuracy Curves')
    plt.xlabel(f'step count')
    plt.ylabel('Accuracy')


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
plot_new_accuracy(('/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/output_one.in', 'Vanilla Transformer'))
plt.show()


