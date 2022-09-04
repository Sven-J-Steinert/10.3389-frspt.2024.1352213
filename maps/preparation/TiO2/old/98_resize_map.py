from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

image = Image.open('WAC_TIO2_GLOBAL_MAP.png')
new_height = 1000
new_image = image.resize((new_height*2, new_height))

new_image.save('WAC_TIO2_RESIZED_MAP.png')
