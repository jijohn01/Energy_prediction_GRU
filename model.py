import torch
import torch.nn as nn
import torch.nn.functional as F

class Sequence(nn.Module):
    def __init__(self,args):
        super(Sequence, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.hidden_size = 200
        self.out_size =3
        self.lstm1 = nn.GRU(input_size=self.input_size,hidden_size=self.hidden_size,batch_first=True)
        self.lstm2 = nn.GRU(input_size=self.hidden_size,hidden_size=self.hidden_size,batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size,self.out_size)


    def forward(self, input,teacher_forcing=True, before_lim=0, after_lim = 0):
        outputs = []
        # input = input.type(dtype=torch.float32)
        if teacher_forcing == True:
            packed_output, h_t = self.lstm1(input)
            packed_output,h_t2 = self.lstm2(packed_output)
            # packed_output, (h_t, c_t) = self.lstm1(input)
            # packed_output, (h_t2, c_t2) = self.lstm2(packed_output)
            outputs=packed_output
            outputs = self.fc1(outputs)
            # outputs = nn.utils.rnn.PackedSequence(outputs1,outputs.batch_sizes)

        else:
            h_t = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            # c_t = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            h_t2 = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            # c_t2 = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()



            for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
                if i<= before_lim-1: #before==30이면 참값은 30개,index는 29번
                    pass
                else: #t일 이후로는 out to input 연결
                    input_t[:, :, 2:5] = outputs[-1]
                    # input_t = torch.cat([input_t[:,:,2:5],outputs[-1]],2)#out to input connection

                o,h_t = self.lstm1(input_t, h_t)
                output,h_t2 = self.lstm2(o, h_t2)
                # o, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))
                # output, (h_t2, c_t2) = self.lstm2(o, (h_t2, c_t2))
                output = self.fc1(output)
                outputs += [output]
            outputs = torch.stack(outputs, 1).squeeze(2)

            # outputs = nn.utils.rnn.PackedSequence(outputs1, outputs.batch_sizes)
        return outputs

class LSTM(nn.Module):
    def __init__(self,args):
        super(LSTM, self).__init__()
        if args.isweather:
            self.input_size = 13
        else:
            self.input_size = 5
        self.hidden_size = 20
        self.out_size =3
        self.lstm1 = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size,self.out_size)


    def forward(self, input,teacher_forcing=True, before_lim=0, after_lim = 0):
        outputs = []
        # input = input.type(dtype=torch.float32)
        if teacher_forcing == True:
            # packed_output, h_t = self.lstm1(input)
            # packed_output,h_t2 = self.lstm2(packed_output)
            packed_output, (h_t, c_t) = self.lstm1(input)
            packed_output, (h_t2, c_t2) = self.lstm2(packed_output)
            outputs=packed_output
            outputs = self.fc1(outputs)
            # outputs = nn.utils.rnn.PackedSequence(outputs1,outputs.batch_sizes)

        else:
            h_t = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            c_t = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            h_t2 = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()
            c_t2 = torch.zeros(1, input.size(0), self.hidden_size, dtype=torch.float).cuda()



            for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
                if i<= before_lim-1: #before==30이면 참값은 30개,index는 29번
                    pass
                else: #t일 이후로는 out to input 연결
                    input_t[:, :, 2:5] = outputs[-1]
                    # input_t = torch.cat([input_t[:,:,2:5],outputs[-1]],2)#out to input connection

                # o,h_t = self.lstm1(input_t, h_t)
                # output,h_t2 = self.lstm2(o, h_t2)
                o, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))
                output, (h_t2, c_t2) = self.lstm2(o, (h_t2, c_t2))
                output = self.fc1(output)
                outputs += [output]
            outputs = torch.stack(outputs, 1).squeeze(2)

            # outputs = nn.utils.rnn.PackedSequence(outputs1, outputs.batch_sizes)
        return outputs