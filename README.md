# SpaceNet Buildings v2

This repository displays the work I did for SpaceNet [Buildings Footprint Extraction Challenge Round 2](https://spacenet.ai/spacenet-buildings-dataset-v2/). The challenge tasked competitors with finding automated methods for **extracting map-ready building footprints** from satellite imagery of 4 different cities. This helps to move toward more accurate, fully automated extraction of buildings aimed at creating better maps where they are most needed.

As a part of this project, I designed a **model training pipeline** using the Fastai Library.

![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Vegas/Residential%203.png)

## Taking a look at the dataset

Though dataset comprises of 4 cities, only Vegas comprises of well clicked noiseless images. The problems with other cities are listed below. 


### Paris
The images are noisy, i.e. out of the 1.1k images only half have buildings in them, rest of the images include only forest cover or agricultural areas. This makes the size of this small dataset even smaller.

The **noisy images** make the training harder and more time consuming as the model has to go through more number of images per epoch. Along with this, the training loss and validation loss we get per epoch are also not correct. This is because these noisy images increase the number of images in the denominator while adding nothing to the loss in the numerator, giving a much lesser loss than the actual value. 

Moreover, the satellite images are off-nadir and dark. **Off-nadir** causes problems on two fronts. Firstly, sides of the buildings are also visible along with the rooftops, but the labels are marked only for the rooftops (that too incorrectly). Secondly, the images are blurred, and the shadows cause problems during segmentation. 

### Khartoum
The dataset is again small with less than 1000 usable images. The dataset also has varying contrast for different areas along with tiny buildings very close to each other which makes it tough for the model to learn. 

### Shanghai
Though the dataset is vast, the images are too dark, slightly off-nadir and lack contrast. 

The clarity in distinguishing between building rooftops, primarily when buildings are clustered together or between buildings and surroundings (trees, roads, etc.) makes the model learn the building boundaries better. Hence the images should have good contrast. 

### What about the Labels?
A problem persists in the labels as well. Whenever there’s a cluster of buildings together, the **labels are also clustered together**. This forces the model to train similarly, i.e., clusters the predicted labels together when buildings are very close to each other. 

Another problem that persists in the labels is that some of **labels are incorrectly marked**. To make some buildings look like a rectangle, they are assumed to extend into trees in the labels. This makes it hard for the model to train, thus giving less dice value. 

## Data Augmentation
Firstly, I removed noisy images from the dataset to make it easier to train and obtain correct metrics and loss values. However, the images removed also take away some of the information about how a farm/forest cover looks like which needs to be included. For this, I added 10% of the removed images to make up for the information lost. 

Next, I moved to Data transforms apart from the default ones that fastai provides. Different brightness and contrast levels were used for different datasets. This is how get_transforms function looks like 

    data = (src.transform(get_transforms(flip_vert=True, max_rotate=15, max_zoom=1.2, max_lighting=0.5,
                          xtra_tfms = [brightness(change=0.56, p=1), contrast(scale=(1, 1.75), p=1.)]), 
                          size = size, tfm_y=True).databunch(bs=bs).normalize(imagenet_stats))

Model for segmentation of Vegas buildings was trained without the need for such extensive data augmentation. The figure below (Shanghai) shows the brightness and contrast adjustment transforms applied to the satellite images to make the distinction of boundaries between two buildings or buildings and background (roads, garden, trees, etc.) more apparent. 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Shanghai/DA%20Shanghai.png)

## Model Pipeline

The model pipeline is built using the Fastai Library. The UNet model uses a **ResNet34 as the encoder** and a peculiar loss function which combines Cross-Entropy loss with Dice Loss. 

The model also uses some tweaks taught in the [fastai course](https://course.fast.ai/) like [one-cycle policy training](https://docs.fast.ai/callbacks.one_cycle.html) per epoch (`fit_one_cycle`), `self-attention=True`, `norm_type=NormType.Weight`, `.to_fp16()` (which helps increasing the batch size by reducing the size of floating point values) and data augmentation mentioned above. 

The [Project Notebooks](https://github.com/the-catalyst/SpaceNet-Buildings-v2/tree/master/Project%20Notebooks) for each dataset walks the reader through the implementation details. 

### Loss Function

The loss function is a **weighed scheme** of `cross-entropy loss` and `dice loss` which provides a suitable dual loss function and is popular in Semantic Segmentation competitions. 

It is observed that cross-entropy loss **optimizes for pixel-level accuracy** whereas the dice loss helps in **improving the segmentation quality/metrics**. Weighted dice loss alone causes over-segmentation at the boundaries, whereas the weighted cross-entropy loss alone leads to very sharp contours with minor under-segmentation. More about this combined dual loss function can be read [in this paper.](https://arxiv.org/pdf/1801.05173.pdf)

### Self Attention
Convolutions in CNN’s primarily work with data that is localized, but this means it can ignore longer range dependencies within the image. The self-attention layer is designed to counter-act this and enforce attention to longer-range dependencies. 

In essence, **attention reweighs certain features of the network** according to some externally or internally (self-attention) supplied weights. This **reweighting of the channel-wise responses** in a particular layer of a CNN by using self-attention helps to model interdependencies between the channels of the convolutional features. 

In the original paper this is based on, they noted that this helps the network **focus more on object shapes rather than local regions** of fixed shapes. This property makes self-attention layers very important in semantic segmentation. 

The following graphs show how Self Attention helps the model converge better and increases the dice value. (Also gives better results on visual inspection.)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Graphs/Self%20Attention%20Graph.png)

## What works?

Different models were trained using the pipeline over various permutations of data augmentation and inclusion/exclusion of noisy images. 

The best results were found after removing the noisy images and augmenting the dataset by adjusting the contrast and brightness as specified in the data augmentation section above. The model with data augmentation and self-attention layers converge to a better dice value faster than the rest combinations. (DAC - Data Augmentation along with Cut “noise free” dataset). 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Graphs/Dice%20Graphs.png)

Note that adding these tweaks makes the Dice value attain higher values, with lesser training. The loss function converges better and is less bumpy. 

## How does it look like?
Two pictures from each city are shown in this section. For detailed results, click on the respective link. 

### Las Vegas
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Vegas/Residential%203.png)
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Vegas/Residential%201.png)
[Vegas results.](https://github.com/the-catalyst/SpaceNet-Buildings-v2/tree/master/Vegas)

### Paris
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Paris/Residential%201.png)
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Paris/Big%20Buildings.png)
[Paris results.](https://github.com/the-catalyst/SpaceNet-Buildings-v2/tree/master/Paris)

### Shanghai
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Shanghai/Residential%201.png)
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Shanghai/Residential%204.png)
[Shanghai results.](https://github.com/the-catalyst/SpaceNet-Buildings-v2/tree/master/Shanghai)

### Khartoum
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Khartoum/Industrial%201.png)
![](https://github.com/the-catalyst/SpaceNet-Buildings-v2/blob/master/Khartoum/Residential%20-%20Housing%201.png)
[Khartoum results.](https://github.com/the-catalyst/SpaceNet-Buildings-v2/tree/master/Khartoum)
