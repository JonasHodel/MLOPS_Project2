import sys
import time
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import datetime

from gluedatamodule import GLUEDataModule
from gluetransformer import GLUETransformer

def get_learning_rate(args):
    arg_value = get_args_value(args, 'learning_rate')
    if arg_value:
        return float(arg_value)
    return float(2e-5) 

def get_warmup_steps(args):
    arg_value = get_args_value(args, 'warmup_steps')
    if arg_value:
        return float(arg_value)
    return 0  

def get_weight_decay(args):
    arg_value = get_args_value(args, 'weight_decay')
    if arg_value:
        return float(arg_value)
    return 0  

def get_args_value(args, key):
    if f'--{key}' in args:
        idx = args.index(f'--{key}')
        return args[idx + 1]
    return None

def print_args(learning_rate, warmup_steps, weight_decay):
    print(f'Learning rate : {learning_rate}')
    print(f'Warmup steps  : {warmup_steps}')
    print(f'Weight decay  : {weight_decay}')   

def get_args_values():
    args = sys.argv
    learning_rate = get_learning_rate(args)
    warmup_steps = get_warmup_steps(args)
    weight_decay = get_weight_decay(args)

    print_args(learning_rate, warmup_steps, weight_decay)

    return learning_rate, warmup_steps, weight_decay

def get_datamodule():
    datamodule = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
    )

    datamodule.setup("fit")

    return datamodule

def get_trainer():
    logger = TensorBoardLogger("tb_logs", name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        logger=logger,
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[lr_monitor]
    )

    return trainer

def get_model(datamanagement, learning_rate, weight_decay, warmup_steps):
    return GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=datamanagement.num_labels,
        eval_splits=datamanagement.eval_splits,
        task_name=datamanagement.task_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        adam_epsilon=1e-8,
        train_batch_size=128,
        eval_batch_size=128,
    )

if __name__ == '__main__':
    seed_everything(42)

    learning_rate, warmup_steps, weight_decay = get_args_values()

    datamodule = get_datamodule()

    model = get_model(datamodule, learning_rate, weight_decay, warmup_steps)

    trainer = get_trainer()

    print("Starting Training...")
    
    start_time = time.time()

    trainer.fit(model, datamodule=datamodule)

    print(f'Finished Training in {time.time() - start_time:.2f} Seconds')
