# A Pytorch object detection framework
**A object detection framework using pytorch**

Now this framework has supported **SSD300** and **RetinaNet**(depth: 18, 34, 50, 102, 151). 

This framework easily enable you to train your own object detetor with multiple popular object detection algorithms. You only need to adjust your data in a specific format, and just wait for your own model ! 



[Dataset Format](#Dataset-Format)
[Commend Line Parameters](#Commend-Line-Parameters)


## Dataset Format

 The data should be in VOC2007 format, which looks like this: 

```python
.
├── Annotations
├── ImageSets
│   └── Main
│       ├── test.txt
│       └── train.txt
└── JPEGImages
```

* In the ```Annotaitons``` dictionary, the xml files which contain the information of the bounding boxes are stored here.

* In the ```JPEGImages``` dictionary, all the images in  train set and test set are stored here

* ```./ImageSets/train.txt``` lists the names of the images in train set, like:

  ```python
  train_image_1.jpg
  train_image_2.jpg
  train_image_3.jpg
  train_image_4.jpg
  train_image_5.jpg
  ```

* ./ImageSets/test.txt``` is similiar with ```./ImageSets/train.txt```, and lists all the image name in test set.

  ```python
  test_image_1.jpg
  test_image_2.jpg
  test_image_3.jpg
  test_image_4.jpg
  test_image_5.jpg
  ```



## Commend Line Parameters

### Train

Running ```train.py``` to train your model. 

```
usage: train.py [-h] {retinanet,ssd} ...

positional arguments:
  {retinanet,ssd}

optional arguments:
  -h, --help       show this help message and exit
```

For example, you want to use **RetinaNet** to implement your object detector, you can see help info by input:

```
python train.py retinanet --help
```



