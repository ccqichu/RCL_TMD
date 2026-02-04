import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import RCLMuFN
from train import train
from data_set import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='RCLMuFN', type=str, help='the model name', choices=['RCLMuFN'])
    parser.add_argument('--text_name', default='text_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels') # 仇恨，非仇恨
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=3e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=3.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.03, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/vis', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--lambda_ratio_start', default=0.0, type=float, help='start weight for CID ratio loss')
    parser.add_argument('--lambda_ratio_end', default=2e-3, type=float, help='end weight for CID ratio loss')
    parser.add_argument('--lambda_itm_start', default=0.0, type=float, help='start weight for CID ITM loss')
    parser.add_argument('--lambda_itm_end', default=1.5e-3, type=float, help='end weight for CID ITM loss')
    parser.add_argument('--lambda_warmup_epochs', default=2, type=int, help='epochs to keep lambda at start value')
    parser.add_argument('--lambda_ramp_epochs', default=5, type=int, help='epochs to ramp from start to end')
    parser.add_argument('--lambda_schedule', default='cosine', type=str,
                        choices=['none', 'linear', 'cosine'], help='schedule type for CID loss weights')
    parser.add_argument('--lambda_chan_entropy', default=0.0, type=float,
                        help='weight for DIMM channel entropy regularization (0 disables)')
    parser.add_argument('--neg_sampling', default='label_aware', type=str,
                        choices=['shuffle', 'label_aware', 'low_sim'],
                        help='negative sampling strategy for CID ITM loss')
    parser.add_argument('--tau_min', default=0.4, type=float,
                        help='minimum temperature for CID softmax')
    parser.add_argument('--tau_decay', default=0.9995, type=float,
                        help='temperature decay rate for CID softmax')
    parser.add_argument('--rho', default=0.3, type=float,
                        help='target ratio for consistent vision patches')
    parser.add_argument('--rho_t', default=0.5, type=float,
                        help='target ratio for consistent text tokens')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='number of heads for DIMM attention')
    parser.add_argument('--cid_smooth_beta', default=5.0, type=float,
                        help='beta for smooth sigmoid scaling in CID masks')
    parser.add_argument('--tau_schedule_mode', default='epoch', type=str,
                        choices=['step', 'epoch'],
                        help='temperature annealing schedule: step-based (per batch) or epoch-based')
    parser.add_argument('--use_ema', action='store_true',
                        help='enable EMA weights for evaluation')
    parser.add_argument('--ema_decay', default=0.9999, type=float,
                        help='EMA decay for model parameters')
    parser.add_argument('--ema_eval_mode', default='ema', type=str,
                        choices=['raw', 'ema', 'both'],
                        help='evaluation mode when EMA is enabled')

    # Two-stage training parameters
    parser.add_argument('--resume_from', default=None, type=str,
                        help='path to checkpoint to resume from (for stage 2 training)')
    parser.add_argument('--freeze_clip', action='store_true',
                        help='freeze CLIP model parameters (for stage 1 training)')


    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if int(args.device) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    seed_everything(args.seed)


    wandb.init(
        project="MMSD2.0",
        notes="mm",
        tags=["mm"],
        config=vars(args),
        mode="offline"
    )

    wandb.watch_called = False  

    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'RCLMuFN':

        processor = CLIPProcessor.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")
        model = RCLMuFN(args)
    else:
        raise RuntimeError('Error model name!')

    # Resume from checkpoint if specified (for stage 2 training)
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            model.load_state_dict(checkpoint)
            print(f"✅ Successfully resumed from {args.resume_from}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

    model.to(device, non_blocking=True)
    wandb.watch(model, log="all")

    train(args, model, device, train_data, dev_data, test_data, processor)



if __name__ == '__main__':
    main()
