import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from src.utils import utility_funcs as uf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class LeNetStandard(nn.Module):
    """
    Standard LeNet CNN architecture
    """
    def __init__(self):
        super(LeNetStandard, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Make a forward pass through the network. Note that here training=true for dropout to ensure that it is not turned off at test time.

        :param x: Observation
        :return: Prediction
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class LeNetDropout(nn.Module):
    """
    LeNet CNN with dropout added into each layer between the pooling and convolutional layer.
    """
    def __init__(self):
        super(LeNetDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Make a forward pass through the network. Note that here training=true for dropout to ensure that it is not turned off at test time.

        :param x: Observation
        :return: Prediction
        """
        x = F.relu(F.max_pool2d(F.dropout(self.conv1(x), training=True), 2))
        x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=True), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x

class KNN:
    def __init__(self, k):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        # One hot enconde label
        enc = LabelEncoder()
        enc.fit(y)
        enc_out = enc.transform(y)
        out = np_utils.to_categorical(enc_out)
        uf.box_print('Fitting KNN with K={}'.format(self.k))
        self.model.fit(X, out)

    def predict(self, X, y=None):
        accuracy = self.model.score(X, y)
        return accuracy