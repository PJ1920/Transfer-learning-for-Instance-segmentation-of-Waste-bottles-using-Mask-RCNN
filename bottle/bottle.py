"""
Mask R-CNN
Train on the custom Bottle dataset and implement color segmentation.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    
    # Train a new model starting from pre-trained bottle weights
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=<path to weight>
    
    eg:
    python bottle/bottle.py train --weights=logs/mask_rcnn_bottle_0100.h5 --dataset=dataset --layer='4+' --aug='Fliprl'
    
    Model Training optional Parameter:
    =================================
    --layer = "'heads' or '4+' or '3+' or 'all' "
    --epoch = " Enter no of epoch for training " default value set as '1'        
    --aug = "'Fliplr' or 'Flipud'" default set to None

    # Apply color segmentation to an image
    python bottle/bottle.py segment --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    
    # Apply color segmentation to bottles for all images in a folder
    python bottle.py segment --weights=/path/to/weights/file.h5 --imagefolder=<path to folder>

    # Apply color segmentation to video using the last weights you trained
    python bottle/bottle.py segment --weights=last --video=<URL or path to file>
    
    eg:
    python bottle/bottle.py segment --weights=logs/mask_rcnn_bottle_0100.h5 --imagefolder=images
    
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import skimage.draw
import imgaug


# Root directory of the project
ROOT_DIR = os.path.abspath("")
print(ROOT_DIR)

#Import Mask_RCNN

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BottleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "bottle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + bottle

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9 # chg as per reqd

    
############################################################
#  Dataset
############################################################

class BottleDataset(utils.Dataset):

    def load_bottle(self, dataset_dir, subset):
        """Load a subset of the Bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("bottle", 1, "bottle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 


            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            #print("Image_Path >", image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "bottle",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "bottle":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bottle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BottleDataset()
    dataset_train.load_bottle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BottleDataset()
    dataset_val.load_bottle(args.dataset, "val")
    dataset_val.prepare()
    
    layers = 'heads'
    epochs=1
    print("Layers test1:",layers,epochs)
    if args.layer==None:
        layers='heads'
        print("Layers testin:",layers,epochs)
    else:
        layers=args.layer
        print("Layers testin1:",layers,epochs)
    
    if args.layer == 'heads':
        layers = 'heads'
        print("Training network heads")        
    else:
        if args.layer == 'all':
            layers = args.layer
            print("Fine tune Resnet",layers,"layers")
        else:
            layers == '3+' or '4+'
            print("Fine tune Resnet stage", layers," and up")
    
    if args.aug == 'Fliplr':
        augmentation = imgaug.augmenters.Fliplr(0.5)
    else: 
        if args.aug == 'Flipud':
            augmentation = imgaug.augmenters.Flipud(0.5)
        else:
            augmentation = None
    epochs=1
    if args.epoch == None:
        epochs=1
    else:
        epochs=int(args.epoch)
        
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
   
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,   # 1 or as per input
                layers=layers,   #'heads' or '3+' or '4+' or 'all'
                augmentation=augmentation) # 'Fliplr' or 'Flipup' or default None 

    
    
######
import cv2
import numpy as np
import os

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        #print("image1",image)
        image=np.array(image)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #print("image2",image)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )    
    return image

def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    It will resize every image to this size before adding them to the video for mp4 output
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
######

def color_segment(image, mask):
    """Apply color segmentation effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        segment = np.where(mask, image, gray).astype(np.uint8)
    else:
        segment = gray.astype(np.uint8)
    return segment


def load_image_into_numpy_array(image):
    # The function supports only grayscale images
    assert len(image.shape) == 2, "Not a grayscale input image" 
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(image, last_axis)
    training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
    assert len(training_image.shape) == 3
    assert training_image.shape[-1] == 3
    return training_image

def detect_and_color_segment(model, image_path=None, video_path=None, imagefolder_path=None):
    assert image_path or video_path or imagefolder_path

    # Image or video?
    ROOT_DIR=os.getcwd()
    class_names=['BG','bottle']
    
    
    if imagefolder_path:
        IMAGE_DIR = os.path.join(ROOT_DIR, args.imagefolder) 
        file_names = next(os.walk(args.imagefolder))[2]
        print(file_names)
        print("DIR",IMAGE_DIR)
        i=0 
        for imgid in file_names:
            print(imgid)              
            image= skimage.io.imread(os.path.join(IMAGE_DIR,imgid ))            

            if image.shape[-1] == 4:
                image = image[..., :3]

            if image.shape[-1] != 3: # for grey scale images
                image=load_image_into_numpy_array(image)


            # Run detection
            results = model.detect([image], verbose=1)
            # Visualize results
            r = results[0]
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # Color segment
            segment = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # Save output
            IMAGEOUT_DIR= os.path.join(ROOT_DIR,"output/")            
            pic=(file_names[i]).split(".")[0]
            file_name =  os.path.join(IMAGEOUT_DIR, pic + "-segment_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
            skimage.io.imsave(file_name, segment)            
            print("Saved to output folder ", file_name)
            i=i+1

    elif image_path:
        IMAGE_DIR = os.path.join(ROOT_DIR, args.image)        
        # Run model detection and generate the color segment effect
        print("Running on {}".format(args.image))
        # Read image        
        image = skimage.io.imread(args.image)
        print("nowin1:",image.shape)
        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.shape[-1] != 3: # for grey scale images
            image = load_image_into_numpy_array(image)
            
        print("nowin:",image.shape)
        
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        
        # Color segment
        segment = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # Save output
        IMAGEOUT_DIR= os.path.join(ROOT_DIR,"output/")
        file_name =  os.path.join(IMAGEOUT_DIR,"outimage_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
        skimage.io.imsave(file_name, segment)         
        print("Saved to output folder ", file_name)     
              
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        
        VIDEOOUT_DIR= os.path.join(ROOT_DIR,"output/")
        
        file_name =  os.path.join(VIDEOOUT_DIR,"outvideo_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))

        # Define codec and create video writer
        #file_name = "outvideo_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()) 
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        frames = []
        frame_count = 0
        print(" Segmentation in progress....")
        while True:
            ret, frame = vcapture.read()
            # Bail out when the video file ends
            if not ret:
                break
                
           
            # Save each frame of the video to a list
            frame_count += 1
            frames.append(frame)
            print('frame_count :{0}'.format(frame_count))
            batch_size=1
            BATCH_SIZE = 1
            if len(frames) == BATCH_SIZE:
                
                results = model.detect(frames, verbose=0)
                print('Predicted')
                for i, item in enumerate(zip(frames, results)):
                    frame = item[0]
                    r = item[1]
                    frame = display_instances(
                        frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                    #name = '{0}.jpg'.format(frame_count + i - batch_size)           
                    #name = os.path.join(VIDEO_SAVE_DIR, name)
                    #cv2.imwrite(name, frame)
                    #print('writing to file:{0}'.format(name))
                    # Clear the frames array to start the next batch
                    frames = []

                # RGB -> BGR to save image to video
                #segment = segment[..., ::-1]
                # Add image to video writer
                vwriter.write(frame)
              
                
        vwriter.release()
        print("... Segmentation Completed!")
        print("Saved to output folder ", file_name)
        
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bottle.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'segment'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/bottle/dataset/",
                        help='Directory of the Bottle dataset')
    
    parser.add_argument('--layer', required=False,
                        metavar="<layer>",
                        help="'heads' or '4+' or '3+' or 'all' ")
    
    parser.add_argument('--epoch', required=False,
                        metavar="<epoch>",
                        help=" Enter noof epoch for training ")
                        
    parser.add_argument('--aug', required=False,
                        metavar="<aug>",
                        help="'Fliplr' or 'Flipud'")
    
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color segmentation')
    parser.add_argument('--imagefolder', required=False,
                        metavar="path or URL to image",
                        help='Images folder to apply the color segmentation')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color segmentation')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "segment":
        assert args.image or args.video or args.imagefolder,\
               "Provide --image or --video or imagefolder to apply color segmentation"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BottleConfig()
    else:
        class InferenceConfig(BottleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0.5
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        layers = 'heads'
        epochs=1
        
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
   # print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "segment":
        detect_and_color_segment(model, image_path=args.image, imagefolder_path=args.imagefolder,
                                video_path=args.video)
        
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segment'".format(args.command))

