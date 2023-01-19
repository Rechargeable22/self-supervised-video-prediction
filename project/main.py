__author__ = 'Jan Scheffczyk, Oliver Leuschner'
__date__ = 'August 2020'

import torch.backends.cudnn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_log
from datetime import datetime

from project.one_frame_model import SingleFrameVideoPredictorModel
from project.one_frame_model_full_res import SingleFrameFullResVideoPredictorModel
from project.utils.toyDataset import *
from project.utils.toyDataset import load_ucf_dataset
torch.random.manual_seed(1)


def main():
    name = "one_frame_fullres"
    c_path = f"model_checkpoints/{name}.chkp"
    # Parameters
    batch_size = 25
    # improve performance for training
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile = False
    torch.autograd.profiler.emit_nvtx = False
    h = 64
    w = 96

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        normalize,
    ])
    print(os.getcwd())

    trainloader, testloader = load_ucf_dataset(batch_size, transform, step_between_clips=1, frame_rate=10,percent=1)
    #trainloader, testloader = toy_dataset(batch_size, transform)
    #trainloader, testloader = moving_ball(batch_size,transform)

    class ValEveryNSteps(pl.Callback):
        def __init__(self, every_n_step):
            self.every_n_step = every_n_step

        def on_batch_end(self, trainer, pl_module):
            if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
                path = c_path
                if os.path.exists(path):
                    os.remove(path)
                trainer.save_checkpoint(path)
                print(torch.cuda.memory_allocated())
                torch.cuda.empty_cache()


    # Model
    model = SingleFrameFullResVideoPredictorModel(h, w)
    if torch.cuda.is_available():
        model.cuda()
    log = pl_log.TensorBoardLogger('runs/' + datetime.now().strftime("%d_%m_%H_%M_")+ name)
    # trainer = pl.Trainer( gpus=1, log_gpu_memory=False, logger=log)
    # c_path = f"model_checkpoints/checkpoint_d.chkp"
    if not os.path.exists(c_path):
        trainer = pl.Trainer(limit_train_batches=10,gpus=1 if torch.cuda.is_available() else 0, log_gpu_memory=False, logger=log, amp_level='O2', benchmark=True,
                             callbacks=[ValEveryNSteps(50)], gradient_clip_val=1,max_epochs=10000)
    else:
        trainer = pl.Trainer(resume_from_checkpoint=c_path, gpus=1 if torch.cuda.is_available() else 0, log_gpu_memory=False, logger=log, amp_level='O2',
                             benchmark=True, callbacks=[ValEveryNSteps(50)], gradient_clip_val=1,max_epochs=10000)

    trainer.fit(model, trainloader, testloader)


if __name__ == '__main__':
    main()
