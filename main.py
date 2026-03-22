import numpy as np
import random

from RNN import RNN
from data import train_data, test_data



#process text training data by placing unique words fromm each sentences key into a set
wordset = set()
for sentence in train_data.keys():
    for word in sentence.split():
        wordset.add(word)

vocab = list(wordset)
vocab_size = len(vocab)
print(vocab)
print(vocab_size)

#map each word to an index
word_to_index = {}
index_to_word  = {}

for i, w in enumerate(vocab):
    word_to_index[w]  = i

for i, w in enumerate(vocab):
    index_to_word[i]  = w

print(word_to_index)
print(index_to_word)

#transform word inputs into 1d vectors
def CreateInputs(sentence):
    inputs = [] 
    for word in sentence.split(' '):
        vector = np.zeros((vocab_size, 1))
        vector[word_to_index[word]] = 1
        inputs.append(vector)
    return inputs

#Applies the Softmax Function to the input array to convert vectors into probaility distribtion
def softmax(xs):
  return np.exp(xs) / sum(np.exp(xs))

#create an instance of our recurrent neural network
rnn = RNN(vocab_size, 2)

#process data
def processData(data, backpropogation=True):
  
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = CreateInputs(x)
    target = int(y)

    # Forward
    output, _ = rnn.feedforward(inputs)
    probabilties = softmax(output)

    # Calculate loss / accuracy
    loss -= np.log(probabilties[target])
    num_correct += int(np.argmax(probabilties) == target)

    if backpropogation:
      # Build dldy
      DL_DY = probabilties
      DL_DY[target] -= 1

      # Backward phase
      rnn.backpropogation(DL_DY)

  return loss / len(data), num_correct / len(data)

#begin training loop
for epoch in range(1050):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (float(train_loss), float(train_acc)))

    test_loss, test_acc = processData(test_data, backpropogation =False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (float(test_loss), float(test_acc)))
