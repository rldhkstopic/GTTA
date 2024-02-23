import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        if self.training:
            noise = torch.rand_like(img) * self.std + self.mean  # for each channel std에 torch.tensor를 추가
            return img + noise
        else:
            return img

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

    
def get_carla_transforms(img_resize, gaussian_std: float=0.005, soft=False):
    size = img_resize
    img_size = (size * 2, size)
    
    base_transforms = [        
        transforms.Resize(img_size),
        transforms.ColorJitter(
            brightness=0.5 if soft else 0.2,
            contrast=0.5 if soft else 0.2,
            saturation=0.5 if soft else 0.2,
            hue=0.1 if soft else 0.05
        ),
        transforms.RandomAffine(
            degrees=10 if soft else 20,
            translate=(0.1, 0.1) if soft else (0.2, 0.2),
            scale=(0.9, 1.1) if soft else (0.8, 1.2),
            shear=5 if soft else 10,
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        GaussianNoise(mean=0., std=gaussian_std),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=InterpolationMode.BILINEAR),
        Clip(min_val=0., max_val=1.),
        # transforms.CenterCrop(img_size)
    ]
    
    return transforms.Compose(base_transforms)

