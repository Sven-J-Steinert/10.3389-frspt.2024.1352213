from PIL import Image, ImageEnhance, ImageOps
Image.MAX_IMAGE_PIXELS = 1000000000

#im = Image.open("LROC_GLOBAL_MARE_360.png")
im = Image.open("LROC_GLOBAL_MARE_180.png")

width, height = im.size
new_width, new_height = (27361,13680)


side_space = int((width - new_width)/2)
print("removing",side_space,"px from both sides")
left = side_space
top = 0
right = width-side_space
bottom = height
im_crop = im.crop((left, top, right, bottom))


im_new = Image.new('L', (new_width, new_height),"white")
x_offset = 0
y_offset = round((new_height-height)/2) + 70 # offset from pdf image
im_new.paste(im_crop, (x_offset,y_offset))

enhancer = ImageEnhance.Contrast(im_new)
im_new = enhancer.enhance(4)

im_invert = ImageOps.invert(im_new)

im_new.save('LROC_GLOBAL_MARE_180_WAC_MASK_NEG.png')
im_invert.save('LROC_GLOBAL_MARE_180_WAC_MASK.png')
print('COMPLETED.')
