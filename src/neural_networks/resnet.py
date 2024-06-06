import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, device):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 2, 1, stride=2)
        self.fc1 = nn.Linear(32, 60)
        self.fc2 = nn.Linear(60, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.device = device
        self.to(self.device)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[0] == 1:
            x = x.reshape((1, 1, 28, 28)).type(torch.float32)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)
        probas = F.softmax(logits, dim=1)
        return probas

    def detailed_forward(self, x):
        x = x.reshape((1, 1, 28, 28)).type(torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def _train_epoch(self, optimizer, criterion, train_loader):
        self.train()
        for _, (data, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data.to(self.device))
            loss = criterion(output.to(self.device), target.to(self.device))
            loss.backward()
            optimizer.step()

    def evaluate(self, criterion, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data.to(self.device))
                test_loss += criterion(output.to(self.device), target.to(self.device)).item()
                pred = output.to("cpu").data.max(1, keepdim=True)[1]
                correct += pred.eq(target.to("cpu").data.view_as(pred)).sum()
                all_preds.extend([i.item() for i in pred])
                all_targets.extend(target.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = (correct / len(test_loader.dataset) * 100).item()
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=1)
        return test_loss, accuracy, precision, recall, f1

    def fit(self, epochs, criterion, optimizer, train_loader):
        for epoch in range(epochs):
            self._train_epoch(optimizer, criterion, train_loader)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, x):
        return self.forward(torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0).to(self.device)).cpu().squeeze().detach().numpy()

    def get_embedding_dim(self):
        return 10

    def update(self, path):
        pass


def load_model(path, device):
    model = ResNet(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
