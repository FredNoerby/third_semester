from PIL import Image
import os
import random
from copy import deepcopy as copy


banana_folder = input('Please paste folder path:\n')
if not(banana_folder.endswith('/')):
    banana_folder += '/'

background = input('\nPlease paste full background path:\n')
# Load image of environment background
background = Image.open(background)

rgb_list = []
filelist = os.listdir(banana_folder) 
for f in filelist: # filelist[:] makes a copy of filelist. 
    if f.startswith("rgb") and f.endswith(".png"): 
        rgb_list.append(f) 
rgb_list.sort() 

for n in rgb_list:
    # Get file number
    file_number = str(n[-9:-3])
    name = str(n.strip('.png'))

    # Load image of rendered banana
    img = Image.open(banana_folder + n)
    # Load image of mask
    mask = Image.open(banana_folder + 'mask' + file_number + 'png')
    # Convert mask image to also have an 'alpha'-channel (Opacity)
    mask = mask.convert('RGBA')
    # Returns the contents of the image as a sequence of pixel values
    datas = mask.getdata()
    # Create an empty list for data sequence
    new_data = []
    # Loop through the data sequence
    for item in datas:
        # Each item (pixel) is a list of four values (R,G,B,A)
        # If R, G, and B == 0 then the pixel is black
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # Append to list with alpha set to zero (100% transparent)
            new_data.append((0, 0, 0, 0))
        else:
            # If the pixel is not black, append as it is
            new_data.append(item)
    # Copies data sequence onto mask picture starting in upper left corner
    mask.putdata(new_data)

    # Create tuple used as offset
    offset = (random.randint(-100,101), random.randint(-100,101))

    # Copy background
    back = copy(background)

    # Paste the transparent banana picture onto the background
    # Third parameter. It indicates a mask that will be used to paste the image. 
    # If you pass a image with transparency, then the alpha channel is used as mask.
    back.paste(img, offset, mask)
    file_name = 'generated'+ file_number + 'jpg'
    back.save('images/' + file_name, 'JPEG')

    with open (banana_folder + 'roi' + file_number + 'txt', 'rt') as in_file:
        xmin, ymin, xmax, ymax = in_file.read().split()
    truncated = 0
    xmin = int(xmin) + offset[0]
    if xmin < 0:
        xmin = 0
        truncated = 1
    ymin = int(ymin) + offset[1]
    if ymin < 0:
        ymin = 0
        truncated = 1
    xmax = int(xmax) + offset[0]
    if xmax > 640:
        xmax = 640
        truncated = 1
    ymax = int(ymax) + offset[1]
    if ymax > 480:
        ymax = 480
        truncated = 1
    with open('annotations/xmls/generated' + file_number + 'xml', 'w') as xml_file: 
        xml_file.write("""<annotation>
    <folder>pictures</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Banana</name>
        <pose>Unspecified</pose>
        <truncated>{}</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
</annotation>""".format(file_name, 'images/' + file_name, truncated, str(xmin), str(ymin), str(xmax), str(ymax)))
