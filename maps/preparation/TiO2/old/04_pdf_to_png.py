import fitz

#pdffile = "LROC_GLOBAL_MARE_360.pdf"
pdffile = "LROC_GLOBAL_MARE_180.pdf"
doc = fitz.open(pdffile)
page = doc.load_page(0)  # number of page
zoom = 20.902    # zoom factor to match 13680 px height after cropping
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix = mat, )
#output = "LROC_GLOBAL_MARE_360.png"
output = "LROC_GLOBAL_MARE_180.png"
pix.save(output)
