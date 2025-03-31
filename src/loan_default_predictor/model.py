import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)





class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # Блок 1
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            # Блок 2
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            # Блок 3
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            # Выход
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)







class BlockWithSkipBN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        out = self.norm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out + x


class DeepNetWithSkipBN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(
            BlockWithSkipBN(hidden_size),
            BlockWithSkipBN(hidden_size),
            BlockWithSkipBN(hidden_size)
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)






class BlockWithDropout(nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        out = self.norm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + x


class DeepNetWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_p=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(
            BlockWithDropout(hidden_size, dropout_p),
            BlockWithDropout(hidden_size, dropout_p),
            BlockWithDropout(hidden_size, dropout_p)
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)