# --------------------------------------------------------
# Based on MAE code bases
# Integrate MAE for Federated Learning
# Reference: https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------
import os
import torch
# 在代码最开头添加
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  # 允许警告但不报错
import math
from typing import Iterable, Optional


from timm.data import Mixup
from timm.utils import accuracy


import sys
sys.path.append(os.path.abspath('..'))
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    proxy_single_client=None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    total_loss = 0.0
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        args.global_step_per_client[proxy_single_client] += 1
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        ###################################
        # 原错误代码
        # with torch.use_deterministic_algorithms(False):
        #     loss_scaler(...)

        # 修改为：
        torch.use_deterministic_algorithms(False, warn_only=True)  # 关闭确定性模式
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        torch.use_deterministic_algorithms(True, warn_only=True)  # 恢复确定性模式（可选）
       #########################################
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 在每个 epoch 结束后可以检查内存使用情况
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#######################################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#######################################################
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    ##########################################################################################
    y_pred = torch.zeros(0, dtype=torch.long, device='cpu')  # []
    y_Prob = torch.zeros(0, dtype=torch.long, device='cpu')
    y_true = torch.zeros(0, dtype=torch.long, device='cpu')
    ####################################################################################

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        ########################################################################
        y_true = torch.cat(
            [y_true, target.view(-1).cpu()])  # [total samples size] #view(-1)表示将tensor转换成一维 #所有的target都拼接在一起
        ########################################################################
        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            #########################################################################
            predicted_probabilities = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)  # dim=1表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
            y_pred = torch.cat([y_pred, preds.view(-1).cpu()])  # 所有的preds都拼接在一起
            y_Prob = torch.cat([y_Prob, predicted_probabilities.cpu()])
            #########################################################################
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
       ##########################################################################################
    num_cls = 37
    from torchmetrics.classification import MulticlassAUROC
    #############################################################################################################################
    # Confusion matrix

    target_names = ['Class-%d' % i for i in range(num_cls)]
    conf_matrix = confusion_matrix(y_true, y_pred)
    # print('classification report')
    # print(classification_report(y_true, y_pred, target_names=target_names))
    report = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
    macro_avg_f1_score = report['macro avg']['f1-score']
    macro_avg_precision = report['macro avg']['precision']
    macro_avg_recall = report['macro avg']['recall']
    print(f"Macro Average F1-Score: {macro_avg_f1_score:.4f}")
    print(f"Macro Average Precision: {macro_avg_precision * 100:.2f}%")
    print(f"Macro Average Recall: {macro_avg_recall * 100:.2f}%")
    metric_macro = MulticlassAUROC(num_classes=num_cls, average="macro", thresholds=None)
    auc_macro = metric_macro(y_Prob, y_true)
    print('Macro Average auc', auc_macro)
    ####################################################################################


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, conf_matrix
