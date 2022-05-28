import numpy as np
import pandas as pd
import os, warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from main import generate_diag
from pytket.extensions.qiskit import AerBackend
from lambeq import AtomicType, IQPAnsatz, TketModel, QuantumTrainer, SPSAOptimizer, Dataset

def read_data(filename):
    labels = []
    df = pd.read_csv(filename, index_col=0)
    for i in range(df.shape[0]):
        l = df.iloc[i]['label']
        t = int(l)
        labels.append([t, 1-t])
    return labels

train_labels = read_data('dataset_144/train_72.csv')
dev_labels = read_data('dataset_144/dev_36.csv')
test_labels = read_data('dataset_144/test_36.csv')

conf = 'spider'

train_diagrams = generate_diag(pd.read_csv('dataset_144/train_72.csv', index_col=0), conf)
dev_diagrams = generate_diag(pd.read_csv('dataset_144/dev_36.csv', index_col=0), conf)
test_diagrams = generate_diag(pd.read_csv('dataset_144/test_36.csv', index_col=0), conf)

ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},n_layers=1, n_single_qubit_params=3)

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

all_circuits = train_circuits+dev_circuits+test_circuits

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

BATCH_SIZE = 36
EPOCHS = 100

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(dev_circuits, dev_labels, shuffle=True)

runs = 6

for i in range(runs):

    print('run: ', i)
    SEED = np.random.randint(100)*i + 99
    print(SEED)

    trainer = QuantumTrainer(
        model,
        loss_function=loss,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        verbose = 'text',
        seed= SEED
    )

    trainer.fit(train_dataset, val_dataset, logging_step=1)

    test_acc = acc(model(test_circuits), test_labels)
    print('Test accuracy:', test_acc)

##############################################################################

# import matplotlib.pyplot as plt

# fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
# ax_tl.set_title('Training set')
# ax_tr.set_title('Development set')
# ax_bl.set_xlabel('Iterations')
# ax_br.set_xlabel('Iterations')
# ax_bl.set_ylabel('Accuracy')
# ax_tl.set_ylabel('Loss')

# colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# ax_tl.plot(trainer.train_epoch_costs, color=next(colours))
# ax_bl.plot(trainer.train_results['acc'], color=next(colours))
# ax_tr.plot(trainer.val_costs, color=next(colours))
# ax_br.plot(trainer.val_results['acc'], color=next(colours))

# test_acc = acc(model(test_circuits), test_labels)
# print('Test accuracy:', test_acc)

# # plt.show()
# plt.savefig('exp1_spider_a005_new_'+str(SEED)+'.png')
