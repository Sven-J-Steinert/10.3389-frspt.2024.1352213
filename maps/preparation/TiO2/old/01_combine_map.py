import sys
from PIL import Image

line1_images = [Image.open(x) for x in ['WAC_TIO2_E350N2250.TIF', 'WAC_TIO2_E350N3150.TIF', 'WAC_TIO2_E350N0450.TIF','WAC_TIO2_E350N1350.TIF']]
line2_images = [Image.open(x) for x in ['WAC_TIO2_E350S2250.TIF', 'WAC_TIO2_E350S3150.TIF', 'WAC_TIO2_E350S0450.TIF','WAC_TIO2_E350S1350.TIF']]

l1_widths, l1_heights = zip(*(i.size for i in line1_images))
l2_widths, l2_heights = zip(*(i.size for i in line2_images))

print('HEIGHTS')
print(l1_heights)
print(l2_heights)
print('WIDTHS')
print(l1_widths)
print(l2_widths)
print()

for i in range (0, len(l1_heights)-1):
    if l1_widths[i] != l1_widths[i+1]:
        print('CRITICAL WARNING     different l1_widths detected')
        print(tuple((l1_widths[i],l1_widths[i+1])))
    if l1_heights[i] != l1_heights[i+1]:
        print('CRITICAL WARNING     different l1_heights detected')
        print(tuple((l1_heights[i],l1_heights[i+1])))

for i in range (0, len(l2_heights)-1):
    if l2_widths[i] != l2_widths[i+1]:
        print('CRITICAL WARNING     different l2_widths detected')
        print(tuple((l2_widths[i],l2_widths[i+1])))
    if l2_heights[i] != l2_heights[i+1]:
        print('CRITICAL WARNING     different l2_heights detected')
        print(tuple((l2_heights[i],l2_heights[i+1])))

if sum(l1_widths) == sum(l2_widths):
    print('SUCCESS  both line widths are matching')

total_height = max(l1_heights) + max(l2_heights)
total_width = max(sum(l1_widths), sum(l2_widths))

new_im = Image.new('L', (total_width, total_height))

x_offset = 0
y_offset = 0
for im in line1_images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

x_offset = 0
y_offset = max(l1_heights)
for im in line2_images:
  new_im.paste(im, (x_offset,y_offset))
  x_offset += im.size[0]

#new_im.save('WAC_TIO2_COMBINED_MAP.png')
new_im.save('WAC_TIO2_COMBINED_MAP.TIF')
print('COMPLETED.')
