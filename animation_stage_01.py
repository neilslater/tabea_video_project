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

# Makes first stage of animation from 0s to 32s, plus 8s overlap with stage 2 for merging

stage_size = 240
num_stages = 4
margin = 60 # This hides rotation artefacts in the corners
frames = stage_size * num_stages
overlap_frames = 240
last_overlap_frame = frames + overlap_frames

# We need 44/45 images of expanding ring, starting from centre to outer edge (1400px)
ring_stages = 45
start_outer_radius = 100
end_outer_radius = 1000
ring_width = 300

# These control start times of circle effects, number of frames offset within each stage_size
first_ring_offset_a = 30
second_ring_offset_a = 45
clear_offset_a = 60

first_ring_offset_b = 120
second_ring_offset_b = 135
clear_offset_b = 150

directory = 'animation_stage_01'
if not os.path.exists(directory):
    os.makedirs(directory)

img0 = PIL.Image.open('images/start_frame_1400x840.jpeg')
img0 = np.float32(img0)

tfi.reset_graph_and_session()

current_img = img0
cropped_img = current_img[margin:-margin, margin:-margin, :]
tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % 0 ) ) )

total_zoom = 1.0
total_rot = 0.0

transition_start = 930
transition_zoom = 0.997
transition_rot = 0.2

targets = [
    'mixed4b_3x3_bottleneck_pre_relu', 105, # girders

    'mixed4a_3x3_bottleneck_pre_relu', 8, # molten glass

    'mixed4a_3x3_bottleneck_pre_relu', 24, # leopard spots

    'mixed3b_3x3_pre_relu', 22, # Glowing croc skin

    'mixed4a_3x3_bottleneck_pre_relu', 83, # paisley

    'mixed4a_3x3_bottleneck_pre_relu', 73, # ball bearings

    'mixed4b_3x3_bottleneck_pre_relu', 68, # fur

    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems

    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems
    'mixed5a_3x3_bottleneck_pre_relu', 3, # Lattice with gems
]

circle_masks = []
ring_masks = []
for ring_frame in range(ring_stages):
    print('Calculating masks, offset {}'.format(ring_frame))
    circle_complete = (1 + ring_frame) / float(ring_stages)
    r_outer = start_outer_radius + circle_complete * ( end_outer_radius - start_outer_radius )
    r_inner = r_outer - ring_width
    if (r_inner < 0):
        circle_img = current_img.copy()
        circle_img[:, :, :] = (0.,0.,0.)
        ring_img = tfi.circle_mask_blurred( current_img, r_outer )
    else:
        circle_img = tfi.circle_mask_blurred( current_img, r_inner )
        ring_img = tfi.ring_mask( current_img, r_outer, r_inner )

    circle_masks.append( circle_img )
    ring_masks.append( ring_img )

    # These are not used, they are just to check the masks are as expected
    tfi.savejpeg( circle_img * 255, ('{}/circle_{}.jpeg'.format( directory, '%04d' % ring_frame ) ) )
    tfi.savejpeg( ring_img * 255, ('{}/ring_{}.jpeg'.format( directory, '%04d' % ring_frame ) ) )


def add_texture_ring( current_img, pattern_step, offset, size, layer, channel ):
    if pattern_step >= offset and pattern_step < offset + size:
        ring_id = pattern_step - offset
        ring_img = ring_masks[ring_id]
        target = tfi.T(layer)[:,:,:,channel]
        textured = tfi.render_deepdream( target, current_img, iter_n=3, step=1.5, octave_n=4, octave_scale=1.5 )
        return tfi.masked_mix( current_img, textured, ring_img )
    else:
        return current_img

def add_clearing_circle( current_img, pattern_step, offset, size, weight = 0.25 ):
    if pattern_step >= offset and pattern_step < offset + size:
        ring_id = pattern_step - offset
        circle_img = circle_masks[ring_id]
        return tfi.masked_mix( current_img, ref_img, circle_img, weight )
    elif pattern_step >= offset + size and pattern_step < offset + size + 4:
        return tfi.mix_images( current_img, ref_img, 1.0 - weight )
    else:
        return current_img

for frame in range(frames):
    fno = frame + 1
    print('Stage 01, frame {}'.format(fno))

    section_id = ( fno // ( stage_size // 2 ) )
    layer = targets[ section_id  * 2 ]
    channel = targets[ section_id * 2 + 1]

    pattern_step = fno % stage_size

    delta_rot = ( 0.02 + section_id * 0.0025 ) * ( 0.5 * math.sin( fno / 23.0 ) + math.sin( fno / 37.0 ) )
    delta_zoom = 1.0 - ( 0.0002 + section_id * 0.000025 ) *  ( 0.5 * math.sin( fno / 17.0 ) + math.sin( fno / 26.0 ) )

    if fno > transition_start:
        rat = (960 - fno)/float(960-transition_start)
        delta_rot = rat * delta_rot  + (1-rat) * transition_rot
        delta_zoom = rat * delta_zoom  + (1-rat) * transition_zoom

    total_rot += delta_rot
    total_zoom *= delta_zoom

    current_img = tfi.affine_zoom( current_img, delta_zoom, delta_rot )
    ref_img = tfi.affine_zoom( img0, total_zoom, total_rot )

    current_img = add_texture_ring( current_img, pattern_step, first_ring_offset_a, ring_stages, layer, channel )
    current_img = add_clearing_circle( current_img, pattern_step, second_ring_offset_a, ring_stages, 0.05 )
    current_img = add_clearing_circle( current_img, pattern_step, clear_offset_a, ring_stages, 0.1 )

    current_img = add_texture_ring( current_img, pattern_step, first_ring_offset_b, ring_stages, layer, channel )

    # The very last section continues without clearing away debris
    if fno < 850:
        current_img = add_clearing_circle( current_img, pattern_step, second_ring_offset_b, ring_stages, 0.05 )
        current_img = add_clearing_circle( current_img, pattern_step, clear_offset_b, ring_stages, 0.1 )

    cropped_img = current_img[margin:-margin, margin:-margin, :]
    tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if (frame % 5 == 0):
        tfi.reset_graph_and_session()


tfi.savejpeg( current_img, ('{}/overlap_frame_{}.jpeg'.format( directory, '%04d' % 960 ) ) )
# Cheating a little, these colours taken from overlap_frame_1200 in stage 02!
end_colours = np.float32( PIL.Image.open('images/stage01_end_colours.jpeg') )

for frame in range(frames,last_overlap_frame):
    fno = frame + 1
    print('Stage 01 - overlap, frame {}'.format(fno))

    section_id = 7
    layer_1 = targets[ section_id  * 2 + 2]
    channel_1 = targets[ section_id * 2 + 3]

    # Lattice with gems, which is what we expect to merge with in stage 2
    layer_2 = 'mixed5a_3x3_bottleneck_pre_relu'
    channel_2 = 3

    if ( fno % 240 ) < 120:
        r = (fno - frames)/120.0
        ri = 1.0 - r
        if layer_1 == layer_2:
            # If the layers match, then shape matches, and we can get a simpler combination
            target = tf.reduce_mean( ri * tfi.T(layer_1)[:,:,:,channel_1] + r * tfi.T(layer_2)[:,:,:,channel_2] )
        else:
            target = ri *  tf.reduce_mean( tfi.T(layer_1)[:,:,:,channel_1] ) + r * tf.reduce_mean( tfi.T(layer_2)[:,:,:,channel_2] )
    else:
        target = tf.reduce_mean( tfi.T(layer_2)[:,:,:,channel_2] )

    delta_rot = transition_rot
    delta_zoom = transition_zoom

    if ( fno % 240 ) > 230:
        delta_rot *= ( 240 - ( fno % 240 ) ) / 10.0

    if fno > 980 and ( fno % 240 ) < 10:
        delta_rot *= ( fno % 240 ) / 10.0

    total_rot += delta_rot
    total_zoom *= delta_zoom

    current_img = tfi.affine_zoom( current_img, delta_zoom, delta_rot )
    ref_img = tfi.affine_zoom( img0, total_zoom, total_rot )
    current_img = tfi.mix_images( current_img, ref_img, 0.998 )
    current_img = tfi.mix_images( current_img, end_colours, 0.99 )
    current_img = tfi.render_deepdream( target, current_img, iter_n=1, step=1.5, octave_n=4, octave_scale=1.5, direct_objective = True )

    tfi.savejpeg( current_img, ('{}/overlap_frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    cropped_img = current_img[margin:-margin, margin:-margin, :]
    tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if (frame % 5 == 0):
        tfi.reset_graph_and_session()

tfi.close_session()