import numpy as np
from numpy.random import randn

def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64, lr=2e-2):
        self.hidden_size = hidden_size
        self.lr = lr

    # embeddings
        self.embedding = xavier_init(input_size, hidden_size)

    # GRU-lite weights
        self.Wz = xavier_init(hidden_size, hidden_size)
        self.Uz = xavier_init(hidden_size, hidden_size)

        self.Wr = xavier_init(hidden_size, hidden_size)
        self.Ur = xavier_init(hidden_size, hidden_size)

        self.Wh = xavier_init(hidden_size, hidden_size)
        self.Uh = xavier_init(hidden_size, hidden_size)

    # output
        self.output_weight = xavier_init(output_size, hidden_size)
        self.output_bias = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        h = np.zeros((self.hidden_size, 1))

        self.inputs_log = inputs
        self.hidden_log = {0: h}

        for word_index in inputs:
            x = self.embedding[word_index].reshape(-1, 1)

            z = self.sigmoid(self.Wz @ x + self.Uz @ h)
            r = self.sigmoid(self.Wr @ x + self.Ur @ h)

            h_tilde = np.tanh(self.Wh @ x + self.Uh @ (r * h))

            h = (1 - z) * h + z * h_tilde

            self.hidden_log[len(self.hidden_log)] = h

        output = self.output_weight @ h + self.output_bias
        return output, h

    def backpropogation(self, d_y):
        n = len(self.hidden_log) - 1

        # gradients
        d_output_weight = d_y @ self.hidden_log[n].T
        d_output_bias = d_y

        d_Wz = np.zeros_like(self.Wz)
        d_Uz = np.zeros_like(self.Uz)
        d_Wr = np.zeros_like(self.Wr)
        d_Ur = np.zeros_like(self.Ur)
        d_Wh = np.zeros_like(self.Wh)
        d_Uh = np.zeros_like(self.Uh)

        # NEW: embedding gradients
        d_embedding = np.zeros_like(self.embedding)

        d_h = self.output_weight.T @ d_y

        for t in reversed(range(1, n + 1)):
            h = self.hidden_log[t]
            h_prev = self.hidden_log[t - 1]

            word_idx = self.inputs_log[t - 1]
            x = self.embedding[word_idx].reshape(-1, 1)

            # recompute gates
            z = self.sigmoid(self.Wz @ x + self.Uz @ h_prev)
            r = self.sigmoid(self.Wr @ x + self.Ur @ h_prev)

            h_tilde = np.tanh(self.Wh @ x + self.Uh @ (r * h_prev))

            # gradients
            dh_tilde = d_h * z
            dz = d_h * (h_tilde - h_prev)

            d_h_tilde_raw = (1 - h_tilde ** 2) * dh_tilde

            d_Wh += d_h_tilde_raw @ x.T
            d_Uh += d_h_tilde_raw @ (r * h_prev).T

            dz_raw = dz * z * (1 - z)
            dr_raw = ((self.Uh.T @ d_h_tilde_raw) * h_prev) * r * (1 - r)

            d_Wz += dz_raw @ x.T
            d_Uz += dz_raw @ h_prev.T

            d_Wr += dr_raw @ x.T
            d_Ur += dr_raw @ h_prev.T

            # embedding gradient update
            d_embedding[word_idx] += (self.Wz.T @ dz_raw +
                                      self.Wr.T @ dr_raw +
                                      self.Wh.T @ d_h_tilde_raw).flatten()

            # backprop hidden
            d_h = (self.Uz.T @ dz_raw +
                   self.Ur.T @ dr_raw +
                   self.Uh.T @ d_h_tilde_raw)

        # gradient clipping (IMPORTANT)
        for d in [d_output_weight, d_output_bias, d_Wz, d_Uz, d_Wr, d_Ur, d_Wh, d_Uh, d_embedding]:
            np.clip(d, -1, 1, out=d)

        # update weights
        self.output_weight -= self.lr * d_output_weight
        self.output_bias -= self.lr * d_output_bias

        self.Wz -= self.lr * d_Wz
        self.Uz -= self.lr * d_Uz

        self.Wr -= self.lr * d_Wr
        self.Ur -= self.lr * d_Ur

        self.Wh -= self.lr * d_Wh
        self.Uh -= self.lr * d_Uh

        # update embeddings
        self.embedding -= self.lr * d_embedding