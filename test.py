import argparse
from tqdm import tqdm

from models.R2Trans import R2Trans
from models.modeling_mask import CONFIGS
from data_utils import get_test_loader
from utils import *

import warnings
import logging

warnings.filterwarnings('ignore')  # ignore annoying warnings!
logging.getLogger().setLevel(logging.ERROR)


def test(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.device = args.device

    if args.dataset == "cub":
        num_classes = 200
    elif args.dataset == "dogs":
        num_classes = 120
    elif args.dataset == "nabirds":
        num_classes = 555

    model = R2Trans(config=config, args=args, num_classes=num_classes)
    if args.checkpoint is not None:
        print('===> load checkpoint: {}'.format(args.checkpoint))
    model.load_state_dict(torch.load(args.checkpoint))
    model = distributed(model, args.device, args.n_gpu)
    num_params = count_parameters(model)

    print("{}".format(config))
    print("Total Parameter: \t%2.1fM" % num_params)

    # Prepare dataset
    test_loader = get_test_loader(args)

    # Validation!
    eval_losses = AverageMeter()
    print("***** Running Validation *****")
    model.eval()
    all_preds, all_label = [], []
    step1_preds, step2_preds, step_concat_preds = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits1, logits2, logits_concat, _ = model(x)
            logits = 4 * logits1 + 4 * logits2 + 2 * logits_concat  # consistent with training settings
            eval_loss = loss_fct(logits/10, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            preds1 = torch.argmax(logits1, dim=-1)
            preds2 = torch.argmax(logits2, dim=-1)
            preds_concat = torch.argmax(logits_concat, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            step1_preds.append(preds1.detach().cpu().numpy())
            step2_preds.append(preds2.detach().cpu().numpy())
            step_concat_preds.append(preds_concat.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            step1_preds[0] = np.append(step1_preds[0], preds1.detach().cpu().numpy(), axis=0)
            step2_preds[0] = np.append(step2_preds[0], preds2.detach().cpu().numpy(), axis=0)
            step_concat_preds[0] = np.append(step_concat_preds[0], preds_concat.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy1 = simple_accuracy(step1_preds, all_label)
    accuracy2 = simple_accuracy(step2_preds, all_label)
    accuracy_concat = simple_accuracy(step_concat_preds, all_label)

    print("Validation Results")
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % accuracy)
    print("Valid Accuracy1: %2.5f" % accuracy1)
    print("Valid Accuracy2: %2.5f" % accuracy2)
    print("Valid Accuracy-concat: %2.5f" % accuracy_concat)
    print("The final acc used in paper is Accuracy1: %2.5f" % accuracy1)
    print("=========================== End ===============================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cub", "dogs", "nabirds"],
                        default="nabirds",
                        help="Which downstream task.")
    parser.add_argument("--data_root", type=str, default='./data/CUB_dataset/CUB_200_2011/')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint", type=str, default='./checkpoint/cub_R2-Trans_checkpoint.pth')

    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=5, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    # Set seed
    set_seed(args)

    test(args)


if __name__ == '__main__':
    main()
