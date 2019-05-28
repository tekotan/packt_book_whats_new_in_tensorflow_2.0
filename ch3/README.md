# Code to illustrate creation of tfrecords 

## images: Create tfrecords from images
### There are three folders, train, validate and test. 
### File names of the images have labels. H*.png is label 1, NH*.png is label 0
These images are taken from ICCAD8 dataset which was available as part of opensource project earlier. <confidentiality??>

## cifar10: Create tfrecords from cifar10 dataset stored in pkl format
To run:
```bash
cd cifar10
python3 create_tfrecords_from_cifar10.py -cf ./cifar-10-batches-py -vidx 4 
```

## Feeding tf.data.Dataset object to a tf.keras model
To run:
```bash
cd cifar10
python3 create_tfrecords_from_cifar10.py -cf ./cifar-10-batches-py -vidx 4 
```

