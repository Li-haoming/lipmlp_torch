import torch
import numpy as np
import torchvision
from torchvision import transforms
import scipy.ndimage

class SignedDistanceTransform:
    def __call__(self, img_tensor):
        # Threshold.
        img_tensor[img_tensor<0.5] = 0.
        img_tensor[img_tensor>=0.5] = 1.

        # Compute signed distances with distance transform
        img_tensor = img_tensor.numpy()

        neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(img_tensor.shape[1])
        signed_distances = torch.Tensor(signed_distances)

        return signed_distances, torch.Tensor(img_tensor)

def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords

class MNISTSDFDataset(torch.utils.data.Dataset):
    def __init__(self, split, size=(256,256)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            SignedDistanceTransform(),
        ])
        self.img_dataset = torchvision.datasets.MNIST('./datasets/MNIST', train=True if split == 'train' else False,
                                                download=True)
        self.meshgrid = get_mgrid(size[0])
        self.im_size = size

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img, digit_class = self.img_dataset[item]

        signed_distance_img, binary_image = self.transform(img)
        
        coord_values = self.meshgrid.reshape(-1, 2)
        signed_distance_values = signed_distance_img.reshape((-1, 1))
        
        indices = torch.randperm(coord_values.shape[0])
        support_indices = indices[:indices.shape[0]//2]
        query_indices = indices[indices.shape[0]//2:]

        meta_dict = {'context': (coord_values[support_indices], signed_distance_values[support_indices]), 'query': (coord_values[query_indices], signed_distance_values[query_indices]), 'all': (coord_values, signed_distance_values)}

        return meta_dict
    
def sdf_loss(predictions, gt, **kwargs):
    return ((predictions - gt)**2).mean()


def inner_maml_sdf_loss(predictions, gt, **kwargs):
    return ((predictions - gt)**2).sum(0).mean()

if __name__ == "__main__":
    train_dataset = MNISTSDFDataset('train', size=(28, 28))
    val_dataset = MNISTSDFDataset('val', size=(28, 28))
    print("train dataset:", train_dataset)
    print("val dataset:", val_dataset)
