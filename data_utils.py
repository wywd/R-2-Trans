from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler

from datasets import CUB, dogs, NABirds


def get_test_loader(args):
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.dataset == 'cub':
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)
    elif args.dataset == 'dogs':
        testset = dogs(root=args.data_root, train=False, cropped=False, transform=test_transform, download=False)
    elif args.dataset == 'nabirds':
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return test_loader
