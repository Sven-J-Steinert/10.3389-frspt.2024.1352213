import geopandas
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def make_image(df, outputname, size=(20, 10), dpi=10000):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    df.plot(ax=ax)
    plt.savefig(outputname, dpi=dpi,  bbox_inches='tight', pad_inches = 0)

#df = geopandas.read_file("LROC_GLOBAL_MARE_360.shp")
df = geopandas.read_file("LROC_GLOBAL_MARE_180.shp")
#boundary = geopandas.GeoSeries(unary_union(df['geometry']))
#df = geopandas.GeoDataFrame({'geometry':boundary})

#make_image(df, 'LROC_GLOBAL_MARE_360.pdf')
make_image(df, 'LROC_GLOBAL_MARE_180.pdf')
