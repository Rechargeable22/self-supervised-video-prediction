import os
from datetime import datetime
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_log

from project.LVLN_model import VideoPredictorModel
from project.one_frame_model import SingleFrameVideoPredictorModel
from project.utils.toyDataset import load_ucf_dataset, toy_dataset
import wandb

class VideoClassifier(LightningModule):
    def __init__(self, image_height, image_width,sgd=True,model=None,hidden_extract=None):
        super(VideoClassifier, self).__init__()
        #self.vpm = SingleFrameVideoPredictorModel(image_height, image_width).load_from_checkpoint("model_checkpoints/adam_0.25.chkp")
        self.vpm = VideoPredictorModel(image_height, image_width).load_from_checkpoint(f"model_checkpoints/adam_0.25.chkp",image_width=image_width,image_height=image_height)
        # self.vpm.load_from_checkpoint("jan")
        for param in self.vpm.parameters():
            param.requires_grad = False
            self.sgd=sgd

        if not model:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(15360, 2000),
                #nn.Linear(25440, 2000),

                nn.BatchNorm1d(2000),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(2000, 101)
            )
        else:
            self.classifier=model
        self.hidden_extract=hidden_extract
        self.criterion = nn.CrossEntropyLoss()
        self.total=0
        self.correct=0
        self.val_steps =0



    def forward(self, x):
        vpm_out = self.vpm(x)   # [torch.Size([8, 3, 128, 160]), torch.Size([8, 3, 128, 160]), torch.Size([8, 3, 128, 160])]



        hidden = self.vpm.hidden # block(4) x gru_n(3) x [bs, channels=64, ]
        hidden=self.hidden_extract(hidden)

        out = self.classifier(hidden)
        return out

    def training_step(self, x, batch_index):
        if torch.cuda.is_available():
            x[0] = x[0].float().cuda()
        out = self(x[0])    # x = torch.Size([bs=10, seq=6, 3, 128, 160])
        loss = self.criterion(out, x[1])
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, x, batch_nb) -> Dict[str, torch.Tensor]:
        if torch.cuda.is_available():
            x[0] = x[0].float().cuda()
        out = self(x[0])    # x = torch.Size([bs=10, seq=6, 3, 128, 160])

        _, predicted = torch.max(out.data, 1)
        self.total += x[1].size(0)
        self.correct += (predicted == x[1]).sum().item()

        loss = self.criterion(out, x[1])
        tensorboard_logs = {'loss':loss,'train_loss': loss}
        wandb.log({"Loss": loss})
        return tensorboard_logs

    def validation_end(self, outputs):
        # ...
        val_avg_acc = 100 * self.correct / self.total
        self.total=0
        self.correct=0
        self.val_steps+=1
        self.logger.experiment.add_scalar('Training/_accuracy', val_avg_acc,self.val_steps)
        print(f"\naccuarcy of {val_avg_acc}%\n")
        wandb.log({"val_accuracy": val_avg_acc})
        return {'Training/_accuracy': val_avg_acc}

    def configure_optimizers(self):
        if self.sgd:
            return torch.optim.SGD([p for p in self.parameters()], lr=1e-2, weight_decay=1e-5,momentum=0.9,dampening=0.1)
        else:
            return torch.optim.AdamW([p for p in self.parameters()], lr=1e-2, weight_decay=1e-5)
def C4(hidden):
    hidden1 = hidden[-1]
    hidden1 = torch.flatten(torch.stack(hidden1, dim=1), start_dim=1)

    return hidden1
def C1_3(hidden):
    hidden2 = [torch.flatten(torch.stack(h, dim=1), start_dim=1) for h in hidden[:-1]]
    hidden2 = torch.cat(hidden2, dim=1)  # torch.Size([8, 3, 64, 8, 10])
    hidden2 = hidden2[:, ::128]
    return hidden2



def run(name="def",optimizer=None,classifier=None,extract=None):
    run = wandb.init(project="gpu2", name=name)


    h = 128
    w = 160


    c_path = f"model_checkpoints/classifier.chkp"

    class ValEveryNSteps(pl.Callback):
        def __init__(self, every_n_step):
            self.every_n_step = every_n_step

        def on_batch_end(self, trainer, pl_module):
            if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
                path = c_path
                if os.path.exists(path):
                    os.remove(path)
                trainer.save_checkpoint(path)
                #print(torch.cuda.memory_allocated())
                torch.cuda.empty_cache()



    #trainloader, testloader = toy_dataset(batch_size, transform)

    #c_path = f"model_checkpoints/checkpoint_adam.chkp"
    # Model
    model = VideoClassifier(h, w,optimizer,classifier,extract)
    model.cuda()
    log = pl_log.TensorBoardLogger('runs/' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_sdg_bn_dp_c4")
    # if os.path.exists(c_path):
    #     trainer = pl.Trainer(resume_from_checkpoint=c_path,gpus=1, logger=log, \
    #                          gradient_clip_val=1, callbacks=[ValEveryNSteps(50)],nb_sanity_val_steps=100,\
    #                          val_check_interval=0.05,val_percent_check=0.03)
    # else:
    trainer = pl.Trainer(gpus=1, logger=log, \
                             gradient_clip_val=1, callbacks=[ValEveryNSteps(50)], nb_sanity_val_steps=5, \
                             val_check_interval=0.03, val_percent_check=0.1,max_epochs=1)

    trainer.fit(model, trainloader, testloader)
    run.finish()


if __name__ == '__main__':
    h = 128
    w = 160
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    batch_size = 20
    trainloader, testloader = load_ucf_dataset(batch_size, transform, step_between_clips=1, frame_rate=10, frames_per_clip=50,percent=0.3)
    # run("sdg+C4+BN+DO",
    #     True,
    #     nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(15360, 2000),
    #         nn.BatchNorm1d(2000),
    #         nn.Dropout(0.5),
    #         nn.ReLU(),
    #         nn.Linear(2000, 101)
    #     )
    #     ,lambda x: torch.cat([C4(x)], dim=1)
    # )
    #
    # run("adam+C4+BN+DO",
    #     False,
    #     nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(15360, 2000),
    #         nn.BatchNorm1d(2000),
    #         nn.Dropout(0.5),
    #         nn.ReLU(),
    #         nn.Linear(2000, 101)
    #     )
    #     , lambda x: torch.cat([C4(x)], dim=1)
    #     )

    # run("sgd+C4,C1-3+BN+DO",
    #     True,
    #     nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(25440, 2000),
    #         nn.BatchNorm1d(2000),
    #         nn.Dropout(0.5),
    #         nn.ReLU(),
    #         nn.Linear(2000, 101)
    #     )
    #     , lambda x: torch.cat([C4(x),C1_3(x)], dim=1)
    #     )

    run("adam+C4,C1-3+BN+DO",
        False,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C4(x), C1_3(x)], dim=1)
        )
    run("sdg+C4,C1-3",
        True,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            #nn.BatchNorm1d(2000),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C4(x), C1_3(x)], dim=1)
        )
    run("sdg+C1-3",
        True,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            #nn.BatchNorm1d(2000),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C1_3(x)], dim=1)
        )
    run("sdg+C4,C1-3+BN",
        True,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            nn.BatchNorm1d(2000),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C4(x), C1_3(x)], dim=1)
        )
    run("sdg+C4,C1-3",
        True,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            #nn.BatchNorm1d(2000),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C4(x), C1_3(x)], dim=1)
        )
    run("sdg+C4,C1-3+DO",
        True,
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(25440, 2000),
            #nn.BatchNorm1d(2000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2000, 101)
        )
        , lambda x: torch.cat([C4(x), C1_3(x)], dim=1)
        )


