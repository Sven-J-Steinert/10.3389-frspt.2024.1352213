import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

image = Image.open("WAC_TIO2_COMBINED_MAP.png")

width, height = image.size
print("width",width,end=" ")
print("height",height,end=" ")

aspect_ratio = width/height
print("aspect_ratio",aspect_ratio)
if aspect_ratio == 2:
    print("aspect ratio already matching.")
    exit(0)
else:
    print("adapting aspect ratio to 2")

if aspect_ratio < 2:
    print("Expanding width")
    print("ERROR:   Not implemented.")
    exit(0)

if aspect_ratio > 2:
    new_height = width/2
    if ((int(new_height) - height)% 2) == 0 :
        new_height = int(new_height)
    else:
        new_height = int(new_height)+1
    print("Expanding height to",new_height)
    add_lines = (new_height-height)/2
    print("adding",add_lines,"lines to the top and bottom")


    new_im = Image.new('L', (width, new_height))
    x_offset = 0
    y_offset = int(add_lines)
    new_im.paste(image, (x_offset,y_offset))

    new_im.save('WAC_TIO2_GLOBAL_MAP.png')
    #new_im.save('WAC_TIO2_GLOBAL_MAP.TIF')
    print('COMPLETED.')
