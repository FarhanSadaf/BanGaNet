import cv2
import numpy as np
import albumentations as A

def preprocess(img, input_size):
    '''
    "Preprocesses an image of word."
    img: Grayscale (H, W)
    Reads image from 'imgpath';
    img (H, W, C) can also be given
        reshapes it according to its ratio;
        fill extra spots by image background; 
        transposes cv2 image.
    Returns transposed image of size 'input_size'.

    Preprocess metodology based in:
        H. Scheidl, S. Fiel and R. Sablatnig,
        Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
        16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
    '''
    wt, ht, ch = input_size

    h, w = img.shape[0], img.shape[1]
    f = max((w / wt), (h / ht))
    new_shape = max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)    # (W, H)

    img = cv2.resize(img, new_shape)    # img.shape : (H, W)

    # As img_shape = (1024, 128, 1) and new img shape = (637, 128)
    # so, we fill remaining spots with 255
    if img.ndim == 2:
        target = np.ones((ht, wt), dtype=np.uint8) * 255
    else:
        target = np.ones((ht, wt, ch), dtype=np.uint8) * 255
    target[:new_shape[1], :new_shape[0]] = img      # target.shape : (input_size[1], input_size[0])

    img = cv2.transpose(target)     # img.shape : (input_size[0], input_size[1])
    return img


def get_augmentation_pipeline(scale_range=(0.9, 1.0), translate_range=(0, 0.05), rotate_range=(-3, 3), shear_range=(-3, 3),
                              sharpen_alpha_range=(0.1, 0.2), blur_limit=3,
                              cutout_holes=4, cutout_max_h=15, cutout_max_w=2, 
                              p=0.6):
    return A.Compose([
        A.Affine(scale=scale_range, translate_percent=translate_range, rotate=rotate_range, shear=shear_range, cval=255, p=0.5),
        A.GaussNoise(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.GridDistortion(p=0.4),
        A.OneOf([
            A.Sharpen(alpha=sharpen_alpha_range, p=1),
            A.Blur(blur_limit=blur_limit, p=1),
            A.MotionBlur(blur_limit=blur_limit, p=1)
        ], p=0.4),
        A.OneOf([
            A.CoarseDropout(num_holes=cutout_holes, max_h_size=cutout_max_h, max_w_size=cutout_max_w, p=1),
            A.CoarseDropout(num_holes=cutout_holes, max_h_size=cutout_max_w, max_w_size=cutout_max_h, p=1)
        ], p=0.4),
    ], p=p)


def augment(aug, imgs):
    '''
    aug: Albumentations augmentation pipeline
    imgs: shape -> (batch_size, width, height, channels); channels if not grayscale
    '''
    return np.array([aug(image=img)['image'] for img in imgs])


def normalization(imgs):
    '''
    Accepts img (1024, 128) : list
    Returns imgs (n, 1024, 128, 1) normalized (1 / 255.0): np.array
    '''
    imgs = np.array(imgs).astype(np.float32)
    imgs = imgs / 255.0
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=-1)
    return imgs


def threshold(img, block_size=25, offset=10):
    '''
    Local gaussian image thresholding.
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset)
