import os
import xml.etree.ElementTree as ET

def generate_csv(root,
                 out_train='annotations.csv',out_test = 'test_annotations.csv',out_classes='classes.csv'):
    train_list = open(os.path.join(root, "ImageSets/Main/train.txt"), 'r').readlines()
    test_list = open(os.path.join(root, "ImageSets/Main/test.txt"), 'r').readlines()
    ann_path = os.path.join(root, "Annotations")
    img_path = os.path.join(root, "JPEGImages")
    classfile = open(out_classes, 'w')
    annotations_train = open(out_train, 'w')
    annotations_test = open(out_test, 'w')
    classes = []
    for index, xml in enumerate(train_list):
        print(xml, str((100 * index) / len(train_list)) + "%")
        tree = ET.parse(os.path.join(ann_path, xml.split("\n")[0] + ".xml"))
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
                classfile.write(classes[-1] + "," + str(classes.index(labelname.text)) + "\n")
            if index==len(train_list)-1:
                write = os.path.join(img_path, xml.split("\n")[
                    0] + ".jpg") + "," + xmin.text + "," + ymin.text + "," + xmax.text + "," + ymax.text + "," + labelname.text
            else:
                write = os.path.join(img_path, xml.split("\n")[
                0] + ".jpg") + "," + xmin.text + "," + ymin.text + "," + xmax.text + "," + ymax.text + "," + labelname.text + "\n"
            annotations_train.write(write)
    annotations_train.flush()
    classfile.flush()

    for index, xml in enumerate(test_list):
        print(xml, str((100 * index) / len(test_list)) + "%")
        tree = ET.parse(os.path.join(ann_path, xml.split("\n")[0] + ".xml"))
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            labelname = obj.find('name')
            box = obj.find('bndbox')
            xmin = box.find('xmin')
            ymin = box.find('ymin')
            xmax = box.find('xmax')
            ymax = box.find('ymax')
            if (index==len(test_list)-1):
                write = os.path.join(img_path,
                                     xml.split("\n")[0] + ".jpg") + "," + xmin.text + "," + ymin.text + "," + xmax.text + "," + ymax.text + "," + labelname.text
            else:
                write = os.path.join(img_path,
                                 xml.split("\n")[0] + ".jpg") + "," + xmin.text + "," + ymin.text + "," + xmax.text + "," + ymax.text + "," + labelname.text + "\n"
            annotations_test.write(write)
    annotations_test.flush()