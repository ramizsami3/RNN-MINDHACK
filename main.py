import numpy as np
import random
from collections import Counter

from RNN import RNN
from data import train_data, test_data

train_data = dict(list(train_data.items())[:500])
test_data = dict(list(test_data.items())[:200])


word_counts = Counter()

for sentence in train_data.keys():
    for word in sentence.split():
        word_counts[word] += 1

MAX_VOCAB = 10000

vocab = [word for word, _ in word_counts.most_common(MAX_VOCAB)]
vocab.append("<UNK>")

word_to_index = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)



def CreateInputs(sentence):
    inputs = []
    for word in sentence.split()[:50]:
        idx = word_to_index.get(word, word_to_index["<UNK>"])
        inputs.append(idx)
    return inputs


def softmax(xs):
    xs = xs - np.max(xs)
    exp = np.exp(xs)
    return exp / np.sum(exp)

rnn = RNN(vocab_size, 2, hidden_size=64, lr=0.0005)


def processData(data, backpropogation=True):
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = CreateInputs(x)
        target = int(y)

        output, _ = rnn.feedforward(inputs)
        probabilties = softmax(output)

        loss -= np.log(probabilties[target] + 1e-8)
        num_correct += int(np.argmax(probabilties) == target)

        if backpropogation:
            DL_DY = probabilties.copy()
            DL_DY[target] -= 1

            rnn.backpropogation(DL_DY)

    return loss / len(data), num_correct / len(data)


for epoch in range(1050):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print(f'--- Epoch {epoch + 1}')
        print(f'Train:\tLoss {train_loss:.3f} | Accuracy: {train_acc:.3f}')

        test_loss, test_acc = processData(test_data, backpropogation=False)
        print(f'Test:\tLoss {test_loss:.3f} | Accuracy: {test_acc:.3f}')
