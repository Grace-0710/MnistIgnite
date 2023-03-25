import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()
    # ArgumentParser : 프로그램을 실행시에 커맨드 라인에 인수를 받아 처리를 간단히 할 수 있도록 하는 표준 라이브러리이다.
    p.add_argument('--model_fn', required=True) # 무조건 필요
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2) # 얼마나 자주 출력을 할까?

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = ImageClassifier(28**2, 10).to(device) # 784개의 데잍를 10개의 class로 분류
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()
    # CrossEntropyLoss : Classification주 사용, 정답 클래스에 해당하는 스코어에 대해서만 로그합을 구하여 최종 Loss
    # 위 모델 softmax로 끝나있음
    # 1. Softmax 함수를 통해 이 값들의 범위는 [0,1], 총 합은 1
    # 2. 1-hot Label (정답 라벨)과의 Cross Entropy를 통해 Loss
   

    trainer = Trainer(config) # ignite 사용항 구현
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config) # main에서 config를 받아!
