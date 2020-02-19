import time
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
from PIL import Image

image1 = Image.open('./frame0.png').convert('RGB')
image2 = Image.open('./frame1.png').convert('RGB')

fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("My Title")

im = ax.imshow( numpy.zeros( ( 256, 256, 3 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

tstart = time.time()
for a in range( 100 ):
    ax.set_title(str( a ) )
    im.set_data(image2)
    im.axes.figure.canvas.draw()

print ( 'FPS:', 100 / ( time.time() - tstart ) )