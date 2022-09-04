import geopandas
from shapely.geometry import Point,Polygon
from shapely.ops import unary_union
from tqdm import tqdm   # progreess bar
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import numpy as np
import matplotlib.pyplot as plt


def t_Lon(Lon):
    if Lon < 0:
        return Lon +360
    if Lon > 0:
        return Lon
    if Lon == 0:
        return 0.000001


def lin_rescale(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


df = geopandas.read_file("LROC_GLOBAL_MARE_360.shp")
boundary = geopandas.GeoSeries(unary_union(df['geometry']))
boundary = geopandas.GeoDataFrame({'geometry':boundary})

#boundary.plot()
#plt.show()

im = Image.open("WAC_TIO2_RESIZED_MAP.png")
data = im.load()
width, height = im.size

num_of_items = width * height
print('initialize',num_of_items)

point_list = [None]*num_of_items
im_point_list = np.zeros(num_of_items, dtype=[('x','<i4'),('y','<i4')])

i = 0
for line in tqdm(range(height)):
    point_y = lin_rescale(line,0,height,90,-90)

    for step in range(width):
        point_x = lin_rescale(step,0,width,-180,180)
        point_xt = t_Lon(point_x)
        point_list[i] = Point(point_xt, point_y)
        im_point_list[i] = (step,line)
        i += 1


points_geoseries = geopandas.GeoSeries(point_list)
points_df = geopandas.GeoDataFrame({'geometry': points_geoseries})

print(points_df)

points_within = geopandas.sjoin(points_df,boundary, op='intersects')
del points_within['index_right']
print(points_df)
print(points_within)

points_outside = points_df.drop(points_within.index)

for i, item in points_outside['geometry'].iteritems():
    data[int(im_point_list[i]['x']),int(im_point_list[i]['y'])] = 22

im.save('WAC_TIO2_RESIZED_MASKED_MAP.png')
#im.save('WAC_TIO2_GLOBAL_MAP.TIF')
print('COMPLETED.')
