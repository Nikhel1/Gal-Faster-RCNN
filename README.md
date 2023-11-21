# Gal-Faster-RCNN
Faster RCNN for Radio Galaxy Detection 

## Installation
Instructions for installation can be found [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Data preparation

Download and extract RadioGalaxyNET data from [here](https://data.csiro.au/collection/61068).
We expect the directory structure to be the following:
```
./RadioGalaxyNET/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
  test/     # test images
```

## Training

To train on a single node with single gpu run:
```
python train.py
```

## Evaluation
To evaluate on test images with a single GPU run:
```
python evaluate.py
```

## License
MIT license.
