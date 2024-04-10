# Assignment 10 details

## Code structure and details

Used following data augmentation techniques using library albumentations. [Source](./datasetProvider.py)
- Random crop. Increased image to 40 and after crop resized to 32x32.
- horizontal flip 
- radom crop using CoarseDropout (max hight, width =8 , p = 0.5)

### Neural network

Network is designed in file: [custom_resnet.py](./custom_resnet.py) with class name ResNet

Every block gets created using class Basic Block which have three fixed layers. Layer configuration changes based on passed parameters. 

1. PrepLayer 
2. Layer-1 and Layer-3 generated using BasicResBlock
    - This blocks also performs Add(X, R1). To add these, resize done by 1x1 with stride = 2 on R1 to match size to X
2. Layer-2: got created by get_custom_conv_with_max_pool_layer
3. Layer-4: Max pooling

### Using One cycle policy

Suggested LR: 7.45E-02


## Results

- Model Parameters count: 5,599,562
- Train accuracy  : 90.62 %
- Test accuracy   : 87.30 %
- Epochs: 24
- Suggested max_lr = 7.45E-02
- initial_lr = max_lr / 100 = 7.45E-04
- LR min = initial_lr / 1000 = 7.45E-07




