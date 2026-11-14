import re
import os
import matplotlib.pyplot as plt

# Path to your log file
log_file_path = './checkpoints/246830_22981040100352_250927-112407/log'

# Lists to store (epoch, loglike) for train, valid, test
train_epochs, train_ll = [], []
valid_epochs, valid_ll = [], []
test_epochs, test_ll = [], []

# Regex patterns to capture each line's epoch and loglike
# Example lines:
#  [ Epoch 96 (train) ]: train loglike is -3.301397499917936, ...
pattern_train = re.compile(r'\[ Epoch (\d+) \(train\) \]: train loglike is ([-\d\.]+)')
pattern_valid = re.compile(r'\[ Epoch (\d+) \(valid\) \]:\s+valid loglike is ([-\d\.]+)')
pattern_test  = re.compile(r'\[ Epoch (\d+) \(test\) \]: test loglike is ([-\d\.]+)')

with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check for train line
        match_tr = pattern_train.search(line)
        if match_tr:
            epoch = int(match_tr.group(1))
            ll = float(match_tr.group(2))
            train_epochs.append(epoch)
            train_ll.append(ll)

        # Check for valid line
        match_va = pattern_valid.search(line)
        if match_va:
            epoch = int(match_va.group(1))
            ll = float(match_va.group(2))
            valid_epochs.append(epoch)
            valid_ll.append(ll)

        # Check for test line
        match_te = pattern_test.search(line)
        if match_te:
            epoch = int(match_te.group(1))
            ll = float(match_te.group(2))
            test_epochs.append(epoch)
            test_ll.append(ll)

method_name = "logloss"
savepath = f'../meeting/new_figures/{method_name}'
os.makedirs(savepath, exist_ok=True)

with plt.style.context({'font.size': 11}):
    plt.figure(figsize=(8, 6))
    plt.plot(train_epochs, train_ll, marker='o', linestyle='-', color='C0', label='Train')  # label='Train'
    plt.plot(valid_epochs, valid_ll, marker='o', linestyle='-', color='C1', label='Valid')  # label='Valid'
    plt.plot(test_epochs, test_ll, marker='o', linestyle='-', color='C2', label='Test')  # label='Test'

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Log-likelihood', fontsize=18)
    #plt.title('Train, Validation, and Test Log-likelihood over Epochs', fontsize=14)
    #plt.ylim(-2.7, -2.54)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight')
    plt.show()
