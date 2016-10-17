###############################################################################################
#   Copyright 2016, Neil Slater.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
###############################################################################################


# Script renders out all channels from selected layers of Inception v5, to help select best ones
# for animations

import tfi
import os
import numpy as np
import PIL.Image
import tensorflow as tf

test_layers = [ 'mixed4a_3x3_bottleneck_pre_relu', 'mixed4b_3x3_bottleneck_pre_relu'
               , 'head1_bottleneck_pre_relu', 'mixed4c_3x3_bottleneck_pre_relu', 'mixed5b_3x3_bottleneck_pre_relu'
               , 'mixed4e_3x3_bottleneck_pre_relu' ]
iterations = 30
source_img = 'images/example.jpg'


img0 = PIL.Image.open( source_img )
img0 = np.float32(img0)

if not os.path.exists('explore_layers'):
        os.makedirs('explore_layers')

for layer in test_layers:
    num_channels = tfi.T(layer).get_shape()[3]
    directory = 'explore_layers/{}'.format( layer )
    if not os.path.exists(directory):
        os.makedirs(directory)

    print( 'Rendering {}, all {} channels squared'.format( layer, num_channels ) )
    test_img = tfi.render_deepdream( tf.square( tfi.T(layer) ), img0, iter_n=iterations, step=2.0, octave_n=4, octave_scale=1.5 )
    tfi.savejpeg( test_img, ('{}/all_channels_squared.jpeg'.format( directory) ) )

    for channel in range(0,num_channels):
        print( 'Rendering {}, channel {}'.format( layer, channel ) )
        test_img = tfi.render_deepdream( tfi.T(layer)[:,:,:,channel], img0, iter_n=iterations, step=2.0, octave_n=4, octave_scale=1.5 )
        tfi.savejpeg( test_img, ('{}/channel_{}.jpeg'.format( directory, '%03d' % channel ) ) )
