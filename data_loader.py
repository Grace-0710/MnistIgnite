import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset): # Pytorch dataset 상속
# Custom Dataset
    def __init__(self, data, labels, flatten=True):
                                    # flatten : 28*28img를 vector로 만들지 여부
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0) # 첫번쨰 dim의 사이즈 == 몇개의 샘플이 있나요?

    def __getitem__(self, idx):
        x = self.data[idx] # x = (28,28)
        y = self.labels[idx]

        if self.flatten:
            x = x.view(-1) # X=(784,)

        return x, y


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y

# mnist는 train / test data 정해져 있음 (train:60000, test:10000)
# train set : 60000장안에서 validation set도 나눠야합니다
def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)
    # |x| = (60000,28,28)
    train_cnt = int(x.size(0) * config.train_ratio) # train 비율에따라
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)) # x.size(0) = 60000
    
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    # |train_x| = (48000,28,28)
    # |valid_x| = (12000,28,28)
    
    train_y, valid_y = torch.index_select( # y -> label도 동일
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True, # train은 무조건 shuffle! ( false시 학습이 잘 안되요! )
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False, # 일부난 보고 싶은 케이스가 있어 주로 false
    )

    return train_loader, valid_loader, test_loader

#위의 애들을 train.py에서 호출합니다
