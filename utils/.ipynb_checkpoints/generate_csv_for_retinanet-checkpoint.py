import os
import xml.etree.ElementTree as ET
train_img_path = '../data/trashV2/train/JPEGImages/'
train_ann_path = '../data/trashV2/train/Annotations/'
test_img_path = '../data/trashV2/test/JPEGImages/'
test_ann_path = '../data/trashV2/test/Annotations/'
classfile = open('../data/trashV2/classes.csv','w')
annotations_train = open('../data/trashV2/annotations.csv','w')
annotations_test = open('../data/trashV2/test_annotations.csv','w')
classes = []
for index,xml in enumerate(os.listdir(train_ann_path)):
    print(xml,str((100*index)/len(os.listdir(train_ann_path)))+"%")
    tree = ET.parse(os.path.join(train_ann_path,xml))
    root = tree.getroot()
    objects = root.findall('object')
    for obj in objects:
        labelname = obj.find('name')
        box = obj.find('bndbox')
        xmin = box.find('xmin')
        ymin = box.find('ymin')
        xmax = box.find('xmax')
        ymax = box.find('ymax')
        if labelname.text not in classes:
            classes.append(labelname.text)
            classfile.write(classes[-1]+","+str(classes.index(labelname.text))+"\n")
        write = os.path.join(train_img_path,xml.split(".")[0]+".jpg")+","+xmin.text+","+ymin.text+","+xmax.text+","+ymax.text+","+labelname.text+"\n"
        annotations_train.write(write)
    annotations_train.flush()
    classfile.flush()

for index, xml in enumerate(os.listdir(test_ann_path)):
    print(xml, str((100 * index) / len(os.listdir(test_ann_path))) + "%")
    tree = ET.parse(os.path.join(test_ann_path, xml))
    root = tree.getroot()
    objects = root.findall('object')
    for obj in objects:
        labelname = obj.find('name')
        box = obj.find('bndbox')
        xmin = box.find('xmin')
        ymin = box.find('ymin')
        xmax = box.find('xmax')
        ymax = box.find('ymax')
        write = os.path.join(test_img_path, xml.split(".")[
            0] + ".jpg") + "," + xmin.text + "," + ymin.text + "," + xmax.text + "," + ymax.text + "," + labelname.text + "\n"
        annotations_test.write(write)
    annotations_test.flush()