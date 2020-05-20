import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self,args):
        super(EncoderLSTM, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50,3)

    def forward(self, input, t=32, end_seq=0):
        outputs = []

        packed_output, hidden1 = self.lstm1(input)  ##여기에서 히든이 몇개나 나오지?
        packed_output, hidden2 = self.lstm2(packed_output)  # 여기서 CUDNN 문제가 발생함
        # outputs, out_lengths = pad_packed_sequence(packed_output, batch_first=True)
        outputs = packed_output
        outputs = self.fc1(outputs)

        return outputs, hidden1, hidden2

class DecoderLSTM(nn.Module):
    def __init__(self,args):
        super(DecoderLSTM, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50,3)

    def forward(self, inputs, hidden1, hidden2, t=32, end_seq=0):
        outputs = []
        packed_output, hidden1 = self.lstm1(inputs, hidden1)
        packed_output, hidden2 = self.lstm2(packed_output, hidden2)
        # outputs, out_lengths = pad_packed_sequence(packed_output, batch_first=True)
        outputs = packed_output
        outputs = self.fc1(outputs)

        return outputs

class EncoderGRU(nn.Module):
    def __init__(self,args):
        super(EncoderGRU, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.lstm1 = nn.GRU(input_size=self.input_size, hidden_size=200, batch_first=True)
        self.lstm2 = nn.GRU(input_size=200, hidden_size=200, batch_first=True)
        self.fc1 = nn.Linear(200,3)

    def forward(self, input, t=32, end_seq=0):
        outputs = []

        packed_output, hidden1 = self.lstm1(input)  ##여기에서 히든이 몇개나 나오지?
        packed_output, hidden2 = self.lstm2(packed_output)  # 여기서 CUDNN 문제가 발생함
        # outputs, out_lengths = pad_packed_sequence(packed_output, batch_first=True)
        outputs = packed_output
        outputs = self.fc1(outputs)

        return outputs, hidden1, hidden2

class DecoderGRU(nn.Module):
    def __init__(self,args):
        super(DecoderGRU, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.lstm1 = nn.GRU(input_size=self.input_size, hidden_size=200, batch_first=True)
        self.lstm2 = nn.GRU(input_size=200, hidden_size=200, batch_first=True)
        self.fc1 = nn.Linear(200,3)

    def forward(self, inputs, hidden1, hidden2, t=32, end_seq=0):
        outputs = []
        packed_output, hidden1 = self.lstm1(inputs, hidden1)
        packed_output, hidden2 = self.lstm2(packed_output, hidden2)
        # outputs, out_lengths = pad_packed_sequence(packed_output, batch_first=True)
        outputs = packed_output
        outputs = self.fc1(outputs)

        return outputs