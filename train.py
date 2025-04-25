# train.py
from main import parse_option
from model import UniMoS
import torch
import os

if __name__ == '__main__':
    args, config = parse_option()
    import os
    from utils.logger import create_logger  # make sure this is at the top of train.py

    # Create output folder
    os.makedirs(config.OUTPUT, exist_ok=True)

    # Create the logger and attach it to args
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}", file=args.file)
    logger.info("Logger initialized for training.")
    args.logger = logger

    torch.cuda.set_device(args.devices)

    trainer = UniMoS(args, config)
    trainer.train()

    # Optionally save the trained model
    model_save_path = os.path.join(config.OUTPUT, f"checkpoint_epoch{trainer.best['epoch']}.pth")
    torch.save({
        'bottleneck': trainer.bottleneck.state_dict(),
        'classifier': trainer.clf.state_dict(),
        'epoch': trainer.best['epoch']
    }, model_save_path)
    print(f"✅ Model saved to {model_save_path}")
