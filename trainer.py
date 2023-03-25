from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine): # 기본 Engine상속받아 custom
    # 기본 param : func == process_function
    # crit, optimizer, config : 추가
    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod # 정적 메서드 : self지정X, 인스턴스의 상태를 변화시키지 않는 메서드를 만들 때 사용
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad() # 한번의 iteration마다 불러오는거라 수행

        x, y = mini_batch # data_loader.py __getitem__ (형태 : 튜플)
        x, y = x.to(engine.device), y.to(engine.device)

        # Take feed-forward -> forward : 입력 텐서로부터 출력 텐서를 계산
        y_hat = engine.model(x) # |x| = (batch size,784)
        #|y|^ = (batch size,10) -> 10차원의 vector : 각 차원의 클래스별 확률값

        loss = engine.crit(y_hat, y)
        loss.backward()
        # backward : 어떤 스칼라 값에 대한 출력 텐서의 변화도(gradient)를 전달받고, 동일한 스칼라 값에 대한 입력 텐서의 변화도를 계산
        # loss.backward() : 전체 그래프는 손실에 대해 미분되며, 그래프 내의 모든 변수는 변화도가 누적된 .grad 함수를 갖게 됩니다.
       
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            # argmax : f(x)의 최대값
            # (torch.argmax(y_hat, dim=-1) == y).sum() : 값이 y인값들의 총 합
            # float(y.size(0)) : batch size
        else:
            accuracy = 0

        # < 학습 보조 지표 >
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        # p_norm 커질수록 파라미터 애들이 많아져 -> 수행중
        g_norm = float(get_grad_norm(engine.model.parameters()))
        # gradient : 학습초기때는 커, 학습이 진행됨에 따라 줄어들어
        # 학습이 진행되도 값이 커지면 불안정하다는 증거 == NaN

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    @staticmethod
    # 학습이 진행되는 과정을 이쁘게 표시하고 싶어요!
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            #EPOCH이 끝났을떄 ( 완료 ) print!
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader
    ):
        # engine 생성하여 실행
        # iteration이 돌떄마다 callback 함수 실행
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.save_model, # function
            train_engine, self.config, # arguments
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
