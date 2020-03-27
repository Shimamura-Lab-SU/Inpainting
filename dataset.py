from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms
import torch

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.photo_path = join(image_dir, "a")
        self.sketch_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]
        #x = 0
        #for dirname in listdir(self.photo_path) 
        #    self.image_filenames[x] = dirname
        #    x = x+1

        #transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)#.type(torch.float16)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)#.type(torch.float16)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

