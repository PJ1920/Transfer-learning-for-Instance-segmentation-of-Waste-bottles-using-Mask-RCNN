 

## Train the bottle model


The code in `bottle.py` is set to train for 1000K steps (1 epochs of 1000 steps each), and using a batch size of 1.

    # Train a new model starting from pre-trained COCO weights
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python bottle.py train --dataset=/path/to/bottle/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python bottle.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    
    # Train a new model starting from pre-trained bottle weights
    python bottle/bottle.py train --dataset=/path/to/bottle/dataset --weights=<path to weight>


    Model Training optional Parameter:
    =================================
    --layer = "'heads' or '4+' or '3+' or 'all' "
    --epoch = " Enter no of epoch for training " default value set as '1'        
    --aug = "'Fliplr' or 'Flipud'" default set to None

eg: ## python bottle/bottle.py train --weights=logs/mask_rcnn_bottle_0100.h5 --dataset=dataset --layer='4+' --aug='Fliprl'
    ## python bottle/bottle.py train --weights=coco --dataset=dataset 


# Apply color segmentation to bottles in an image

    # Apply color segmentation to bottles in an image
    python bottle/bottle.py segment --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color segmentation to bottles for all images in a folder
    python bottle.py segment --weights=/path/to/weights/file.h5 --imagefolder=<path to folder>

    # Apply color segmentation to bottles video using the last weights you trained
    python bottle.py segment --weights=last --video=<URL or path to file>


    --image = Image file path to apply the color segmentation
    --imagefolder = Images folder name to apply the color segmentation
    --video = Video file path to apply the color segmentation
                                                            
eg: ## python bottle/bottle.py segment --weights=logs/mask_rcnn_bottle_0100.h5 --image=images/junktest2.jpg
    ## python bottle/bottle.py segment --weights=logs/mask_rcnn_bottle_0100.h5 --imagefolder=images
    ## python bottle/bottle.py segment --weights=logs/mask_rcnn_bottle_0100.h5 --video=videos/plasticbottle.mov
