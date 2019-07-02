
# coding: utf-8

# ## 舰船参考
# 
# function []=Get_Local_img(MeanPos,bow_angle,length,width)
# 
# width=width/2; %舰宽
# 
# Bow_x=MeanPos(1,1);
# Bow_y=MeanPos(2,1);
# 
# Tail_x=MeanPos(1,1)+length*cos(bow_angle);
# Tail_y=MeanPos(2,1)-length*sin(bow_angle);
# %Local_img=gray(MeanPos_y(1)-n/4*sin(MeanThea):MeanPos_y(1)+10*sin(MeanThea),MeanPos_x(1)-10*cos(MeanThea):MeanPos_x(1)+n/4*cos(MeanThea));
# 
#     %Bow A- B  
# BowA_x=round(Bow_x+width*cos(pi/2+bow_angle)); 
# BowA_y=round(Bow_y-width*sin(pi/2+bow_angle));
# 
# BowB_x=round(Bow_x-width*cos(pi/2+bow_angle));
# BowB_y=round(Bow_y+width*sin(pi/2+bow_angle));
#     %Tail A- B 
# TailA_x=round(Tail_x+width*cos(pi/2+bow_angle));
# TailA_y=round(Tail_y-width*sin(pi/2+bow_angle));
# 
# TailB_x=round(Tail_x-width*cos(pi/2+bow_angle));
# TailB_y=round(Tail_y+width*sin(pi/2+bow_angle));
# 
# hold on
# plot([BowA_x,BowB_x,TailB_x,TailA_x,BowA_x],[BowA_y,BowB_y,TailB_y,TailA_y,BowA_y],'r','LineWidth',0.3)


import os  
import xml.etree.ElementTree as ET
import math
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import glob

origin_ann_dir = '/home/jzchen/data/RemoteSensing/ships/HRSC2016/HRSC2016/Train/Annotations/'
img_dir = '/home/jzchen/data/RemoteSensing/ships/HRSC2016/HRSC2016/Train/AllImages/'  
new_ann_dir = '/home/jzchen/WorkingSpace/R2CNN_HEAD_FPN_Tensorflow/Annotations/'
pi = 3.141592


#解析文件名出来

xml_Lists = glob.glob(origin_ann_dir + '/*.xml')
len(xml_Lists)

xml_basenames = []
for item in xml_Lists:
    xml_basenames.append(os.path.basename(item))

xml_names = []
for item in xml_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_path = img_dir + temp1 + '.bmp'
    if os.path.exists(img_path):
        xml_names.append(temp1)
    else:
        print(temp1 + '.xml')

for it in xml_names:        
    tree = ET.parse(os.path.join(origin_ann_dir,str(it)+'.xml'))
    root = tree.getroot()

    #HRSC_Objects=root.findall('HRSC_Objects')
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = '/HRSC2016/FullDataSet/AllImages/'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(it)+'.bmp' # str(1) + '.jpg'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = root.find('Img_SizeWidth').text
    node_height = SubElement(node_size, 'height')
    node_height.text = root.find('Img_SizeHeight').text
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = root.find('Img_SizeDepth').text

    Objects = root.findall('./HRSC_Objects/HRSC_Object')
    if len(Objects) == 0:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'back_ground'
    else:
        for Object in Objects:
            name = Object.find('Class_ID').text
            if name in ['100000001','100000002','100000003','100000004']:
                continue
            mbox_cx = float(Object.find('mbox_cx').text)
            mbox_cy = float(Object.find('mbox_cy').text)
            mbox_w = float(Object.find('mbox_w').text)
            mbox_h = float(Object.find('mbox_h').text)
            mbox_ang = float(Object.find('mbox_ang').text)
            # print(mbox_cx,mbox_cy,mbox_w,mbox_h,mbox_ang)

            #计算舰首 与舰尾点坐标

            bow_x = mbox_cx+mbox_w/2*math.cos(mbox_ang)
            bow_y = mbox_cy+mbox_w/2*math.sin(mbox_ang)

            tail_x = mbox_cx-mbox_w/2*math.cos(mbox_ang)
            tail_y = mbox_cy-mbox_w/2*math.sin(mbox_ang)

            # print(bow_x,bow_y,tail_x,tail_y)
            
            
            bowA_x = round(bow_x+mbox_h/2*math.sin(mbox_ang))
            bowA_y = round(bow_y-mbox_h/2*math.cos(mbox_ang))

            bowB_x = round(bow_x-mbox_h/2*math.sin(mbox_ang))
            bowB_y = round(bow_y+mbox_h/2*math.cos(mbox_ang))


            tailA_x = round(tail_x+mbox_h/2*math.sin(mbox_ang))
            tailA_y = round(tail_y-mbox_h/2*math.cos(mbox_ang))

            tailB_x = round(tail_x-mbox_h/2*math.sin(mbox_ang))
            tailB_y = round(tail_y+mbox_h/2*math.cos(mbox_ang))

            # print(bow_x,bow_y,tail_x,tail_y)
            # print(bowA_x,bowA_y,bowB_x,bowB_y,tailA_x,tailA_y,tailB_x,tailB_y)
            node_object = SubElement(node_root, 'object')

            node_name = SubElement(node_object, 'name')
            node_name.text = name

            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')

            node_x1 = SubElement(node_bndbox, 'x1')
            node_x1.text = str(bowA_x)
            node_y1 = SubElement(node_bndbox, 'y1')
            node_y1.text = str(bowA_y)

            node_x2 = SubElement(node_bndbox, 'x2')
            node_x2.text = str(bowB_x)
            node_y2 = SubElement(node_bndbox, 'y2')
            node_y2.text = str(bowB_y)



            node_x3 = SubElement(node_bndbox, 'x3')
            node_x3.text = str(tailA_x)
            node_y3 = SubElement(node_bndbox, 'y3')
            node_y3.text = str(tailA_y)

            node_x4 = SubElement(node_bndbox, 'x4')
            node_x4.text = str(tailB_x)
            node_y4 = SubElement(node_bndbox, 'y4')
            node_y4.text = str(tailB_y)

            node_header_x = SubElement(node_bndbox, 'header_x')
            node_header_x.text = str(Object.find('header_x').text)
            node_header_y = SubElement(node_bndbox, 'header_y')
            node_header_y.text = str(Object.find('header_y').text)    

            # node_xmin = SubElement(node_bndbox, 'xmin')
            # node_xmin.text = str(min(bowA_x,bowB_x,tailA_x,tailB_x))
            # node_ymin = SubElement(node_bndbox, 'ymin')
            # node_ymin.text = str(min(bowA_y,bowB_y,tailA_y,tailB_y))
            # node_xmax = SubElement(node_bndbox, 'xmax')
            # node_xmax.text = str(max(bowA_x,bowB_x,tailA_x,tailB_x))
            # node_ymax = SubElement(node_bndbox, 'ymax')
            # node_ymax.text = str(max(bowA_y,bowB_y,tailA_y,tailB_y))

            xml = tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
            dom = parseString(xml)
            fw = open(os.path.join(new_ann_dir,str(it)+'.xml'), 'wb')
            fw.write(xml)
            # print("xml _ ok")
            fw.close()    
    
    
    

