import math, copy, time
import logging
from abc import abstractmethod, abstractclassmethod

import torch
from torch import nn, optim

logger = logging.getLogger(__name__)


class McBaseAttentionModel(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, train_batch):
        """메인 모델의 forward 메서드입니다. model(input) 을 통해 실행 합니다.
        모델에 필요한 인풋들을 train_batch 파라미터로 받아서 실행하고
        모델의 ouput을 리턴합니다. 상속클래스에서 반드시 구현해야합니다. """
        raise NotImplementedError("forward function is needed")

    @classmethod
    def collate_fn(cls, batch):
        """batch 클래스에서 메인 모델의 input과 gt 형태로 바꾸어 주는 함수입니다. 이 함수의 output은
        model의 input으로 사용하는 pytorch tensor형태여야 합니다. 반드시 구현해야합니다. """
        raise NotImplementedError("Collate function is needed. (batch-> actual input)")

    @classmethod
    def train_on_epoch(cls, cur_epoch, train_data_iterator, model, loss_func, optimizer, verbose=False):
        """각 모델에 맞게 배치를 적절히 변환해서 학습을 진행시키는 Trainer입니다.
        일단 급한대로 함수로 구현하고 추후에 Trainer 클래스로 바꾸고 공통된 부분은 부모나 공통함수로 빼면 되겠습니다. """

        start = time.time()
        total_tokens = 0
        total_loss = 0

        model.train()
        for i, batch in enumerate(train_data_iterator):
            train_batch_input, train_batch_gt = cls.collate_fn(batch)

            # ******************* train! *********************
            out, attn_vals = model(train_batch_input)
            #     print("Final_out:", out, "Gt:", train_batch_gt)
            loss_value = loss_func(out, train_batch_gt)
            #     print("loss:", loss_val)
            total_loss += loss_value
            total_tokens += len(train_batch_input[0])

            loss_value.backward()
            optimizer.step()
            # **************************************************

        # total_train_loss = total_loss / total_tokens
        total_train_loss = total_loss

        # Some validation here
        elapsed = time.time() - start
        logger.info(f"Epoch Step: {cur_epoch} Train Loss: {total_train_loss} elapsed: {elapsed}")
        return {
            "total_n_batch": i,
            "total_train_loss_by_token": total_train_loss,
            "total_tokens": total_tokens
        }

    @classmethod
    def validation_on_epoch(cls, cur_epoch, model, val_generator, val_metrics, verbose=False):
        """한 epoch이 끝났을때 val_data_loader를 받아서 validation을 진행합니다. """
        start = time.time()
        result = dict()
        total_size = 0
        total_val_pred = []
        total_val_gt = []
        model.eval()
        for i, batch in enumerate(val_generator):
            val_input, val_gt = cls.collate_fn(batch)
            val_pred, attn_val = model(val_input)
            total_val_pred.append(val_pred)
            total_val_gt.append(val_gt)
            total_size += len(batch)
        assert total_val_pred != [], "No validation data."

        total_val_pred = torch.concat(total_val_pred)  # concat all prediction
        total_val_gt = torch.concat(total_val_gt)  # concat all ground truths

        for key, metric_func in val_metrics.items():
            result[key] = metric_func(total_val_pred.to('cpu').detach(), total_val_gt.to('cpu').detach())

        elapsed_time = time.time() - start
        logger.info(f"cur_epoch:{cur_epoch}, validation elapsed_time:{elapsed_time}")
        return result

    def copy(self):
        return copy.deepcopy(self)
