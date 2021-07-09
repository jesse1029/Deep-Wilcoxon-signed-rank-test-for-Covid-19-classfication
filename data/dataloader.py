import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A

from .dataset import Covid19

def get_loader(args):
    
    transform = A.Compose([
                        A.RandomRotate90(),
                        A.Resize(224,224),
                        #A.VerticalFlip(),
                        #A.HorizontalFlip(),
                       # A.Transpose(),
                        A.OneOf([
                            A.IAAAdditiveGaussianNoise(),
                            A.GaussNoise(),
                            A.GaussianBlur(),
                        ], p=0.3),
                        A.OneOf([
                            A.MotionBlur(p=.2),
                            A.MedianBlur(blur_limit=3, p=.1),
                            A.Blur(blur_limit=3, p=.1),
                        ], p=0.2),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
                        A.OneOf([
                            A.OpticalDistortion(p=0.3),
                            A.GridDistortion(p=.1),
                            A.IAAPiecewiseAffine(p=0.3),
                        ], p=0.2),
                        A.OneOf([
                            A.CLAHE(clip_limit=2),
                            A.IAASharpen(),
                            A.IAAEmboss(),
                            A.RandomContrast(),
                            A.RandomBrightness(),
                            ], p=0.3),
                        ], p=1)
    
    train_data_pos = Covid19('train/pos', args.path, args.train_batch, transform = transform)
    train_data_neg = Covid19('train/neg', args.path, args.train_batch, transform = transform)
    
    val_data_pos = Covid19('val/pos', args.path, args.val_batch, transform = transform)
    val_data_neg = Covid19('val/neg', args.path, args.val_batch, transform = transform)
    
    train_loader_pos = DataLoader(dataset = train_data_pos, batch_size = args.train_ct_batch, shuffle=True, num_workers=16)
    train_loader_neg = DataLoader(dataset = train_data_neg, batch_size = args.train_ct_batch, shuffle=True, num_workers=16)

    train_loaders = {'pos': train_loader_pos,
                     'neg': train_loader_neg}
    
    val_loader_pos = DataLoader(dataset = val_data_pos, batch_size = 1, shuffle=False, num_workers=8)
    val_loader_neg = DataLoader(dataset = val_data_neg, batch_size = 1, shuffle=False, num_workers=8)
    
    val_loaders = {'pos': val_loader_pos,
                   'neg': val_loader_neg}
    
    return train_loaders, val_loaders