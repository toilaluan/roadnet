import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
from torchvision import transforms as T
import albumentations as A
import imgaug.augmenters as iaa
import random
color_augment_instance = iaa.OneOf([
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Grayscale(alpha=(0.0, 0.5)),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        iaa.RemoveSaturation((0.0, 0.5))
    ])

contrast_augment_instance = iaa.OneOf([
    iaa.GammaContrast((0.5, 2.0)),
    iaa.SigmoidContrast(gain=(3, 5), cutoff=(0.2, 0.4)),
    iaa.LinearContrast((0.4, 1.6))
])
blur_augment_instance = iaa.OneOf([
    iaa.GaussianBlur(sigma=1.5),
    iaa.AverageBlur(k=3),
    iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),

])
arithmetic_instance = iaa.OneOf([
    iaa.AddElementwise((-20, 20)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
    iaa.AdditivePoissonNoise(lam=(0, 10)),
    iaa.Multiply((0.7, 1.2)),
    iaa.MultiplyElementwise((0.7, 1.2)),
    iaa.ImpulseNoise(0.05),
    iaa.SaltAndPepper(0.05),
    iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
])

augmenters = [color_augment_instance, contrast_augment_instance, blur_augment_instance, arithmetic_instance]
class RoadDataset(Dataset):
    def __init__(self, folder_path, size = 448) -> None:
        super().__init__()
        self.size = size
        self.samples = glob.glob(folder_path + '/*')
        # img_size = np.random.randint(scale)
        self.transforms = A.Compose([
            # A.RandomResizedCrop(size, size, (0.1,0.2)),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
        ])
        self.tensor_transforms = T.Compose([
            T.ToTensor(),
            # T.Normalize((0.5), (0.5))
        ])
        print(self.samples)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        """
        Parameters:
            index - a integer for indexing sample
        Return a dict of:
            image (tensor) : a road image, n_channels = 3
            segment (tensor) : segment of road in image, n_channels = 1
            center_line (tensor) : center line of road in image, n_channels = 1
            edge (tensor) : edge of road in image, n_channels = 1
        """
        folder_path = self.samples[index]
        folder_id = folder_path.split('/')[-1]
        img_path = folder_path + '/Ottawa-{}.tif'.format(folder_id)
        center_line_path = folder_path + '/centerline.png'
        edge_path = folder_path + '/edge.png'
        segment_path = folder_path + '/segmentation.png'

        image    = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        segment = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED)
        center_line = cv2.imread(center_line_path, cv2.IMREAD_UNCHANGED)
        edge = cv2.imread(edge_path, cv2.IMREAD_UNCHANGED)

        segment = self.convert_to_binary_image(segment)
        center_line = self.convert_to_binary_image(center_line)
        edge = self.convert_to_binary_image(edge)

        masks = [segment, center_line, edge]
        transformed = self.transforms(image = image, masks = masks)
        image = transformed['image']
        auger = random.choice(augmenters)
        image = auger(image = image)
        image = self.tensor_transforms(image)
        segment, center_line, edge = transformed['masks']
        # cv2.imwrite("segment.png", segment*255)
        segment = cv2.resize(segment, (self.size // 4, self.size // 4))
        segment = torch.from_numpy(segment).unsqueeze(0).float()
        center_line = torch.from_numpy(center_line).unsqueeze(0).float()
        edge = torch.from_numpy(edge).unsqueeze(0).float()        
        return {
            'image' : image,
            'segment' : segment,
            'center_line' : center_line,
            'edge' : edge
        }
        

    def convert_to_binary_image(self, img):
        img = np.array(img == 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
if __name__ == "__main__":
    road_dataset = RoadDataset('Ottawa-Dataset')
    sample = road_dataset[0]
    image = sample['image']
    segment = sample['segment']
    print(segment)
    print(segment.shape)
    image = np.array(image.permute(1,2,0)*255).astype(np.uint8)
    label = np.array(segment.permute(1,2,0)*255).astype(np.uint8)
    label = cv2.resize(label, (448,448))
    # debug = cv2.hconcat(image, label)
    # print(debug.shape)
    # plt.imshow(image.permute(1,2,0))
    cv2.imwrite("debug.png", label)
    cv2.imwrite("input.png", image)