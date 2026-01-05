import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    parser.add_argument('--device', default='3', type=str, help='device number')
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
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lambda_ratio_start', default=0.0, type=float, help='start weight for CID ratio loss')
    parser.add_argument('--lambda_ratio_end', default=2e-3, type=float, help='end weight for CID ratio loss')
    parser.add_argument('--lambda_itm_start', default=0.0, type=float, help='start weight for CID ITM loss')
    parser.add_argument('--lambda_itm_end', default=1.5e-3, type=float, help='end weight for CID ITM loss')
    parser.add_argument('--lambda_warmup_epochs', default=2, type=int, help='epochs to keep lambda at start value')
    parser.add_argument('--lambda_ramp_epochs', default=3, type=int, help='epochs to ramp from start to end')
    parser.add_argument('--lambda_schedule', default='linear', type=str,
                        choices=['none', 'linear', 'cosine'], help='schedule type for CID loss weights')
    parser.add_argument('--neg_sampling', default='label_aware', type=str,
                        choices=['shuffle', 'label_aware', 'low_sim'],
                        help='negative sampling strategy for CID ITM loss')
    parser.add_argument('--tau_schedule_mode', default='epoch', type=str,
                        choices=['step', 'epoch'],
                        help='temperature annealing schedule: step-based (per batch) or epoch-based')

    # Two-stage training parameters
    parser.add_argument('--resume_from', default=None, type=str,
                        help='path to checkpoint to resume from (for stage 2 training)')
    parser.add_argument('--freeze_clip', action='store_true',
                        help='freeze CLIP model parameters (for stage 1 training)')
    # Counterfactual (CF) + prediction consistency
    parser.add_argument('--use_cf', action='store_true',
                        help='enable counterfactual intervention on border/layout cues (representation-level)')
    parser.add_argument('--cf_prob', default=1.0, type=float,
                        help='probability to apply CF per batch (0~1)')
    parser.add_argument('--cf_border_width', default=1, type=int,
                        help='border ring width on ViT patch grid (e.g., 1 for outer ring)')
    parser.add_argument('--cf_replace', default='mean_con', type=str,
                        choices=['zero', 'mean_con', 'noise'],
                        help='how to replace border-inconsistent tokens in CF')

    parser.add_argument('--pc_loss', default='kl', type=str, choices=['kl', 'mse'],
                        help='prediction consistency loss type')
    parser.add_argument('--pc_tau', default=1.0, type=float,
                        help='temperature for KL consistency')
    parser.add_argument('--no_pc_detach_teacher', action='store_false',
                        dest='pc_detach_teacher', default=True,
                        help='disable stop-gradient on teacher/original branch for KL')

    # weight scheduling for prediction consistency
    parser.add_argument('--lambda_pc_start', default=0.0, type=float,
                        help='start weight for prediction consistency regularizer')
    parser.add_argument('--lambda_pc_end', default=1e-3, type=float,
                        help='end weight for prediction consistency regularizer')
    parser.add_argument('--lambda_pc_warmup_epochs', default=2, type=int,
                        help='epochs to keep lambda_pc at start value')
    parser.add_argument('--lambda_pc_ramp_epochs', default=3, type=int,
                        help='epochs to ramp lambda_pc from start to end')
    parser.add_argument('--lambda_pc_schedule', default='linear', type=str,
                        choices=['none', 'linear', 'cosine'],
                        help='schedule type for lambda_pc')


    # * Backbone (disabled)
    # parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--backbone', default='resnet50', type=str,
    #                     help="Name of the convolutional backbone to use")
    # parser.add_argument('--dilation', action='store_true',
    #                     help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    #                     help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--hidden_dim', default=256, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")
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

    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

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
