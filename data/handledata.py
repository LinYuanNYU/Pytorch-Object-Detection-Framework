import os
datadir = "trashV2/JPEGImages"
xmldir = "trashV2/Annotations"
traindir = "trashV2/train/JPEGImages"
trainxmldir = "trashV2/train/Annotations"
testdir = "trashV2/test/JPEGImages"
testxml = "trashV2/test/Annotations"
count = 0
for item in os.listdir(datadir):
    if count%5==0:
        print(os.path.join(datadir,item),os.path.join(testdir,item))
        #os.rename(os.path.join(datadir,item),os.path.join(testdir,item))
        xml = item.split(".")[0]+".xml"
        print(os.path.join(xmldir, xml), os.path.join(testxml, xml))
        #os.rename(os.path.join(xmldir,xml),os.path.join(testxml,xml))
