import sys
sys.path.append('./python')
sys.path.append('./apps')
from models import LanguageModel
from simple_training import *
import needle.nn as nn
import needle as ndl
import itertools
import torch
import pytest
import numpy as np
# import mugrade


np.random.seed(3)


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
                                    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ['tanh', 'relu']


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn_cell(batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
    if init_hidden:
        h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        h_ = model_(torch.tensor(x), None)

    model = nn.RNNCell(input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity)
    model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)
    if init_hidden:
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        h = model(ndl.Tensor(x, device=device), None)
    assert h.device == device
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(),
                               model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm_cell(batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    if init_hidden:
        h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        h_, c_ = model_(torch.tensor(x), None)

    model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

    model.W_ih = nn.Parameter(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = nn.Parameter(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = nn.Parameter(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = nn.Parameter(model_.bias_hh.detach().numpy(), device=device)

    if init_hidden:
        h, c = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        h, c = model(ndl.Tensor(x, device=device), None)
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(),
                               model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)


SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, nonlinearity=nonlinearity)
    if init_hidden:
        output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        output_, h_ = model_(torch.tensor(x), None)

    model = nn.RNN(input_size, hidden_size, num_layers, bias, device=device, nonlinearity=nonlinearity)
    for k in range(num_layers):
        model.rnn_cells[k].W_ih = ndl.Tensor(
            getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.rnn_cells[k].W_hh = ndl.Tensor(
            getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.rnn_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.rnn_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        output, h = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(
    ), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
    if init_hidden:
        output_, (h_, c_) = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        output_, (h_, c_) = model_(torch.tensor(x), None)

    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
    for k in range(num_layers):
        model.lstm_cells[k].W_ih = ndl.Tensor(
            getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.lstm_cells[k].W_hh = ndl.Tensor(
            getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.lstm_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.lstm_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, (h, c) = model(ndl.Tensor(x, device=device),
                               (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        output, (h, c) = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.lstm_cells[0].W_ih.grad.detach().numpy(
    ), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


OUTPUT_SIZES = [1, 1000]
EMBEDDING_SIZES = [1, 34]
SEQ_MODEL = ['rnn', 'lstm']


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("embedding_size", EMBEDDING_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("output_size", OUTPUT_SIZES)
@pytest.mark.parametrize("seq_model", SEQ_MODEL)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_language_model_implementation(seq_length, num_layers, batch_size, embedding_size, hidden_size,
                                       init_hidden, output_size, seq_model, device):
    # TODO add test for just nn.embedding?
    x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
    h0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
    c0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)

    model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
    if init_hidden:
        if seq_model == 'lstm':
            h = (h0, c0)
        elif seq_model == 'rnn':
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)
    else:
        output, h_ = model(ndl.Tensor(x, device=device), None)

    if seq_model == 'lstm':
        assert isinstance(h_, tuple)
        h0_, c0_ = h_
        assert c0_.shape == (num_layers, batch_size, hidden_size)
    elif seq_model == 'rnn':
        h0_ = h_
    assert h0_.shape == (num_layers, batch_size, hidden_size)
    assert output.shape == (batch_size * seq_length, output_size)
    # TODO actually test values
    output.backward()
    for name, p in model.named_parameters().items():
        assert getattr(p, "grad", False) or not init_hidden and name.endswith("W_hh")


### PTB training ###
def epoch_general_ptb_torch(data, model, seq_len=40, loss_fn=torch.nn.CrossEntropyLoss(), opt=None,
                            clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    correct, loss_sum, n_step, n_samplers = 0., 0., 0, 0
    if opt:
        model.train()
    else:
        model.eval()

    h = None
    for i in range(0, data.shape[0]-1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device, dtype)
        X, y = torch.Tensor(X.numpy()).type(torch.long), torch.Tensor(y.numpy())
        if opt:
            opt.zero_grad()
        # NOTE: use
        pred, h = model(X, h)
        print("torch", pred)
        if isinstance(h, tuple):
            h = (h[0].detach(), h[1].detach())
        else:
            h = h.detach()

        loss = loss_fn(pred, y.type(torch.long))
        correct += (pred.detach().numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        # NOTE multiply seq_len
        loss_sum += loss.detach().numpy() * y.shape[0]
        n_step += 1
        n_samplers += y.shape[0]

    return correct / n_samplers, loss_sum / n_samplers
    # END YOUR SOLUTION


def train_ptb_torch(model, data, seq_len=40, n_epochs=1, optimizer=torch.optim.SGD,
                    lr=4.0, weight_decay=0.0, loss_fn=torch.nn.CrossEntropyLoss, clip=None,
                    device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        train_acc, train_loss = epoch_general_ptb_torch(
            data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
        # print(train_acc, train_loss)
    return train_acc, train_loss
    # END YOUR SOLUTION


def evaluate_ptb_torch(model, data, seq_len=40, loss_fn=torch.nn.CrossEntropyLoss,
                       device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    return epoch_general_ptb_torch(data, model, seq_len=seq_len, loss_fn=loss_fn(), device=device, dtype=dtype)
    # END YOUR SOLUTION


class TorchLanguageModel(torch.nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn'):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(TorchLanguageModel, self).__init__()
        # BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size, embedding_size)
        if seq_model == 'rnn':
            self.model = torch.nn.RNN(embedding_size, hidden_size, num_layers)
        elif seq_model == 'lstm':
            self.model = torch.nn.LSTM(embedding_size, hidden_size, num_layers)
        else:
            raise NotImplementedError()
        self.linear = torch.nn.Linear(hidden_size, output_size)
        # END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        # BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x = self.embedding(x)   # (seq_len, bs, embedding_size)
        x, h_ = self.model(x, h)  # (seq_len, bs, hidden_size)
        x = self.linear(x.reshape((seq_len*bs, self.hidden_size)))
        return x, h_
        # END YOUR SOLUTION


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_language_model_training(device):
    np.random.seed(0)
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 10
    num_examples = 100
    batch_size = 16
    seq_model = 'rnn'
    num_layers = 2
    hidden_size = 10
    n_epochs = 2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
    model_ = TorchLanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size,
                                num_layers=num_layers, seq_model=seq_model)
    model = LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size,
                          num_layers=num_layers, seq_model=seq_model, device=device)
    for name, param in model_.named_parameters():
        if param.requires_grad:
            print(name)
    model.embedding.weight = nn.Parameter(
        model_.get_parameter('embedding.weight').detach().numpy(), device=device)
    np.testing.assert_allclose(model_.get_parameter('embedding.weight').detach().numpy().sum(),
                               model.embedding.weight.numpy().sum(), atol=1e-5, rtol=1e-5)
    for k in range(num_layers):
        model.model.rnn_cells[k].W_ih = nn.Parameter(
            model_.get_parameter(f'model.weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.model.rnn_cells[k].W_hh = nn.Parameter(
            model_.get_parameter(f'model.weight_hh_l{k}').detach().numpy().transpose(), device=device)
        # if bias:
        model.model.rnn_cells[k].bias_ih = nn.Parameter(
            model_.get_parameter(f'model.bias_ih_l{k}').detach().numpy(), device=device)
        model.model.rnn_cells[k].bias_hh = nn.Parameter(
            model_.get_parameter(f'model.bias_hh_l{k}').detach().numpy(), device=device)
    model.linear.weight = nn.Parameter(
        model_.get_parameter('linear.weight').detach().numpy().transpose(), device=device)
    model.linear.bias = nn.Parameter(
        model_.get_parameter('linear.bias').detach().numpy().reshape((1, 381)), device=device)
    train_acc_, train_loss_ = train_ptb_torch(model_, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc_, test_loss_ = evaluate_ptb_torch(model_, train_data, seq_len=seq_len, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    if str(device) == "cpu()":
        np.testing.assert_allclose(train_loss_, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(test_loss_, test_loss, atol=1e-5, rtol=1e-5)
        # np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
        # np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)
    elif str(device) == "cuda()":
        np.testing.assert_allclose(train_loss_, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(test_loss_, test_loss, atol=1e-5, rtol=1e-5)
        # np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
        # np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)


### MUGRADE ###
TEST_BATCH_SIZES = [6]
TEST_INPUT_SIZES = [3]
TEST_HIDDEN_SIZES = [5]
TEST_SEQ_LENGTHS = [7]
TEST_NUM_LAYERS = [3]
TEST_OUTPUT_SIZES = [16]
TEST_EMBEDDING_SIZES = [8]
TEST_SEQ_MODEL = ['rnn', 'lstm']


def mugrade_submit(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()[:64]
        # print(x)
        mugrade.submit(x)
    else:
        # print(x)
        mugrade.submit(x)


def submit_rnn():
    # devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]

    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')

    for (device, batch_size, input_size, hidden_size) in itertools.product(
            devices, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        model = nn.RNNCell(input_size, hidden_size, device=device)
        mugrade_submit(model.W_ih.numpy())
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
        mugrade_submit(h.numpy())
        h.sum().backward()
        mugrade_submit(model.W_hh.grad.numpy())

    for (device, seq_length, num_layers, batch_size, input_size, hidden_size) in itertools.product(
            devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        model = nn.RNN(input_size, hidden_size, num_layers, device=device)
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
        mugrade_submit(h.numpy())
        mugrade_submit(output.numpy())
        output.sum().backward()
        mugrade_submit(model.rnn_cells[-1].W_hh.grad.numpy())


def submit_lstm():
    # devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]
    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')
    for (device, batch_size, input_size, hidden_size) in itertools.product(
            devices, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        model = nn.LSTMCell(input_size, hidden_size, device=device)
        mugrade_submit(model.W_hh.numpy())
        (h, c) = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
        mugrade_submit(h.numpy())
        mugrade_submit(c.numpy())
        h.sum().backward()
        mugrade_submit(model.W_hh.grad.numpy())

    for (device, seq_length, num_layers, batch_size, input_size, hidden_size) in itertools.product(
            devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        model = nn.LSTM(input_size, hidden_size, num_layers, device=device)
        output, (h, c) = model(ndl.Tensor(x, device=device),
                               (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
        mugrade_submit(h.numpy())
        mugrade_submit(c.numpy())
        mugrade_submit(output.numpy())
        output.sum().backward()
        mugrade_submit(model.lstm_cells[-1].W_hh.grad.numpy())


def submit_language_model():
    # devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]
    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')
    for (device, seq_length, num_layers, batch_size, embedding_size, hidden_size, seq_model, output_size) in itertools.product(
            devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_EMBEDDING_SIZES, TEST_HIDDEN_SIZES, TEST_SEQ_MODEL, TEST_OUTPUT_SIZES):
        x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
        h0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
        c0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
        model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
        if seq_model == 'lstm':
            h = (h0, c0)
        elif seq_model == 'rnn':
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)
        if seq_model == 'lstm':
            h0_, c0_ = h_
            mugrade_submit(c0_.numpy())
        elif seq_model == 'rnn':
            h0_ = h_
        mugrade_submit(h0_.numpy())
        mugrade_submit(output.numpy())

    device = ndl.cpu()  # TODO CHANGE BACK
    # device = ndl.cpu()
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 8
    num_examples = 88
    batch_size = 12
    seq_model = 'lstm'
    num_layers = 2
    hidden_size = 12
    n_epochs = 2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
    model = LanguageModel(28, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers,
                          seq_model=seq_model, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    mugrade_submit(train_loss)
    mugrade_submit(test_loss)


if __name__ == "__main__":
    submit_rnn()
    submit_lstm()
    submit_language_model()
