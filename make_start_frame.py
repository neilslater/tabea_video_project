import tfi
import os
import numpy as np
import PIL.Image
import tensorflow as tf

in_name = 'nfrac_1400x840.jpg'
out_name = 'start_frame_1400x840.jpeg'

img0 = PIL.Image.open(in_name)
img0 = np.float32(img0)

tfi.reset_graph_and_session()

target = tf.square( tfi.T('mixed4c') )

print( 'Rendering {}'.format( out_name ) )
test_img = tfi.render_deepdream( target, img0, iter_n=20, step=0.75, octave_n=4, octave_scale=1.5 )
tfi.savejpeg( test_img, out_name )
