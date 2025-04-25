# test.py
from utils.logger import create_logger
import os
import re

from main import parse_option
from model import UniMoS
import torch
import os

if __name__ == '__main__':
    args, config = parse_option()
    # adding codes from chat
    # Set up logger (same as train.py)
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}", file=args.file)
    logger.info("Starting testing...")
    args.logger = logger  # ✅ Fixes the AttributeError
    # codes form chat ends here
    torch.cuda.set_device(args.devices)

    trainer = UniMoS(args, config)

    # Path to the checkpoint you want to test
    # i will replace model_path = ... with the code right below it
    # model_path = os.path.join(config.OUTPUT, f"checkpoint_epoch{trainer.best['epoch']}.pth")

    # 🔍 Auto-find best model in output folder
    model_files = [f for f in os.listdir(config.OUTPUT) if f.startswith("best_model") and f.endswith(".pth")]
    if not model_files:
        print("❌ No saved model found! Make sure to train first.")
        exit()

      # Sort alphabetically (so epoch25 > epoch9)
    # Sort by epoch number extracted from the filename
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1


    model_files.sort(key=extract_epoch)
    model_path = os.path.join(config.OUTPUT, model_files[-1])
    print(f"🧠 Loading model: {model_path}")
    model_path = os.path.join(config.OUTPUT, model_files[-1])




    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        trainer.bottleneck.load_state_dict(checkpoint['bottleneck'])
        trainer.clf.load_state_dict(checkpoint['classifier'])
        trainer.epoch = checkpoint['epoch']
        print(f"✅ Loaded model from epoch {trainer.epoch}")
    else:
        print("❌ No saved model found! Train first with train.py.")
        exit()

    # Run testing
    trainer.test()
    print(f"🧠 Loaded model path: {model_path}")

