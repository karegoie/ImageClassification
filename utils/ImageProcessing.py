from torchvision import transforms


'''
def center_crop(im):
    width, height, depth = im.shape  # Get dimensions

    if depth !=3:
        depth, width, height = im.shape

    si = int(np.round(img_size/2))

    left = int(np.round(width / 2 - si))
    right = int(np.round(width / 2 + si))
    top = int(np.round(height / 2 + si))
    bottom = int(np.round(height / 2 - si))

    # Crop the center of the image
    im_cropped = im[bottom:top, left:right, :]

    return im_cropped
'''

class ImageTransform():
    def __init__(self, args):
        self.args = args
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.args.img_size, self.args.img_size)), # transforms.Resize(args.img_size) # transforms.CenterCrop(args.img_size)
            # transforms.GaussianBlur(kernel_size=11),
            # transforms.AutoAugment(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.data_transform(img)

'''
something more for utils
'''
