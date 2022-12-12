from model import LitMNIST

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch

    
def main(): 
    model = LitMNIST()

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)
    trainer.test()
    torch.save(model.state_dict(), "model.pt")
    
if __name__ == "__main__":
    main()
