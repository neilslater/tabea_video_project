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

import tfi
import os
import numpy as np
import PIL.Image
import tensorflow as tf
import math

# Each entry corresponds to 240 frames = 8 seconds of video
targets = [
    # Previous
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems ++
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems ++
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems ++
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems ++

    # Quieter bit
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems ++
    'head1_bottleneck_pre_relu', 62, # Ocean pattern ++
    'head0_bottleneck_pre_relu', 116, # Pointy ++
    'mixed4b_3x3_bottleneck_pre_relu', 28, # fuzzy links ++

    # Added bass drums
    'head0_bottleneck_pre_relu', 3, # Harbour/islands ++
    'head0_bottleneck_pre_relu', 16, # Baubles ++
    'head0_bottleneck_pre_relu', 22, # Arches ++
    'head0_bottleneck_pre_relu', 67, # Crystal wings ++

    # Louder bit
    'head0_bottleneck_pre_relu', 84, # Garden ruins
    'mixed4b_3x3_bottleneck_pre_relu', 110, # patterned ++
    'mixed5a_3x3_bottleneck_pre_relu', 76, # structured swirls ++
    'head0_bottleneck_pre_relu', 0, # Domed buildings ++

    'head0_bottleneck_pre_relu', 114, # Tiger ++
    'head0_bottleneck_pre_relu', 124, # Glowing doors ++
    'mixed4b_3x3_bottleneck_pre_relu', 111, # geigery ++
    'mixed4b_3x3_bottleneck_pre_relu', 111, # geigery ++

    # Spares
    'head1_bottleneck_pre_relu', 64, # Birds
    'mixed3a_3x3_pre_relu', 9999, # tf.square( all )
    'mixed4a_3x3_bottleneck_pre_relu', 42, # worms
    'mixed4b_3x3_bottleneck_pre_relu', 68, # fur
    'head0_bottleneck_pre_relu', 18, # Trumpets
    'head0_bottleneck_pre_relu', 23, # Eye waves?
    'head0_bottleneck_pre_relu', 26, # Network
    'head0_bottleneck_pre_relu', 47, # Pyramids
    'head0_bottleneck_pre_relu', 53, # Feathers
    'head0_bottleneck_pre_relu', 127, # Bead circles
    'head0_bottleneck_pre_relu', 120, # Snakes
    'head0_bottleneck_pre_relu', 116, # Pointy
    'head1_bottleneck_pre_relu', 45, # Odd machinery
    'head1_bottleneck_pre_relu', 59, # Ocean pattern
    'head1_bottleneck_pre_relu', 65, # Turtles
    'head1_bottleneck_pre_relu', 93, # Little buildings
    'head1_bottleneck_pre_relu', 108, # Firey patches
    'head1_bottleneck_pre_relu', 125, # Appliances
]

channel_step = 240 # 16 targets
start_frame = 960
end_frame = 4800
nframes = end_frame - start_frame
margin = 60 # This hides rotation artefacts in the corners

directory = 'animation_stage_02'
if not os.path.exists(directory):
    os.makedirs(directory)

# Technically this is the end frame, as we're working backwards towards it
img0 = PIL.Image.open('images/start_frame_1400x840.jpeg')
img0 = np.float32(img0)
colour_guides = [
    img0,
    np.float32( PIL.Image.open('images/colour_guide_a.jpeg') ),
    np.float32( PIL.Image.open('images/colour_guide_b.jpeg') ),
    np.float32( PIL.Image.open('images/colour_guide_c.jpeg') )
]

tfi.reset_graph_and_session()

current_img = img0.copy()
cropped_img = current_img[margin:-margin, margin:-margin, :]
tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % end_frame ) ) )

slow_zoom = 1.0/0.997
slow_rot = 0.2
fast_zoom = 1.0/0.994
fast_rot = 0.35

for frame in range(nframes):
    fno = end_frame - 1 - frame
    section_id = ( fno // channel_step )

    prev_layer = targets[ (section_id-1)  * 2 ]
    prev_channel = targets[ (section_id-1) * 2 + 1]

    layer = targets[ section_id  * 2 ]
    channel = targets[ section_id * 2 + 1]

    print( 'Rendering frame {}, using layer {}, channel {}'.format( fno, layer, channel ) )

    if channel > 1000:
        target = tf.square( tfi.T(layer) )
    else:
        target = tfi.T(layer)[:,:,:,channel]

    # Mixed target for first half of each channel_step
    if ( fno % 240 ) < 120:
        r = (fno % 240)/120.0
        ri = 1.0 - r
        if prev_layer == layer:
            # If the layers match, then shape matches, and we can get a simpler combination
            target = tf.reduce_mean( ri * tfi.T(prev_layer)[:,:,:,prev_channel] + r * tfi.T(layer)[:,:,:,channel] )
        else:
            target = ri *  tf.reduce_mean( tfi.T(prev_layer)[:,:,:,prev_channel] ) + r * tf.reduce_mean( tfi.T(layer)[:,:,:,channel] )
    else:
        target = tf.reduce_mean( tfi.T(layer)[:,:,:,channel] )

    rot = slow_rot
    zoom = slow_zoom

    # We start with section 4
    if section_id >= 8:
        rot = fast_rot
        zoom = fast_zoom
    elif section_id == 7:
        rot = 0.75 * fast_rot + 0.25 * slow_rot
        zoom = 0.75 * fast_zoom + 0.25 * slow_zoom
    elif section_id == 6:
        rot = 0.5 * fast_rot + 0.5 * slow_rot
        zoom = 0.5 * fast_zoom + 0.5 * slow_zoom
    elif section_id == 5:
        rot = 0.25 * fast_rot + 0.75 * slow_rot
        zoom = 0.25 * fast_zoom + 0.75 * slow_zoom

    # This should line up with 960 being reverse of same thing in stage_01, which we want!
    if ( fno % 480 ) < 240:
        rot = -rot

    if ( fno > 970 ):
        if ( fno % 240 ) < 10:
            rot *= ( fno % 240 )/10.0

        if ( fno % 240 ) > 230:
            rot *= ( 240 - ( fno % 240 ) ) / 10.0

    step_val = 1.25
    if (fno > 4600):
        step_val = 0.5 + 0.75 * (4800 - fno)/200.0

    current_img = tfi.mix_images( current_img, colour_guides[ section_id % 4 ], 0.997 )
    current_img = tfi.affine_zoom( current_img, zoom, rot )
    current_img = tfi.render_deepdream( target, current_img, iter_n=1, step=step_val, octave_n=4, octave_scale=1.5, direct_objective = True )
    cropped_img = current_img[margin:-margin, margin:-margin, :]
    tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if ( fno < 1201 ):
        tfi.savejpeg( current_img, ('{}/overlap_frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if (frame % 5 == 0):
        tfi.reset_graph_and_session()

tfi.close_session()