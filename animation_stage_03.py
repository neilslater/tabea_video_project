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

margin = 60 # This hides rotation artefacts in the corners
start_frame = 4800
end_frame = 5040

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

directory = 'animation_stage_03'
if not os.path.exists(directory):
    os.makedirs(directory)

img0 = PIL.Image.open('images/start_frame_1400x840.jpeg')
img0 = np.float32(img0)

tfi.reset_graph_and_session()

current_img = img0
cropped_img = current_img[margin:-margin, margin:-margin, :]
tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % start_frame ) ) )

total_zoom = 1.0
total_rot = 0.0

targets = [
    'head0_bottleneck_pre_relu', 53, # Feathers
    'mixed5a_3x3_bottleneck_pre_relu', 11, # Dog face and circles
    'mixed5a_3x3_bottleneck_pre_relu', 119, # Butterfly
    'mixed5a_3x3_bottleneck_pre_relu', 33, # Spider monkey brains
    'head0_bottleneck_pre_relu', 18, # Trumpets
    'head0_bottleneck_pre_relu', 23, # Eye waves
    'head0_bottleneck_pre_relu', 26, # Network
    'head0_bottleneck_pre_relu', 47, # Pyramids
    'head0_bottleneck_pre_relu', 127, # Bead circles
    'head0_bottleneck_pre_relu', 124, # Glowing doors
    'head0_bottleneck_pre_relu', 120, # Snakes
    'head0_bottleneck_pre_relu', 116, # Pointy
    'head1_bottleneck_pre_relu', 45, # Odd machinery
    'head1_bottleneck_pre_relu', 59, # Ocean pattern
    'head1_bottleneck_pre_relu', 65, # Turtles
    'head1_bottleneck_pre_relu', 93, # Little buildings
    'head1_bottleneck_pre_relu', 108, # Firey patches
    'head1_bottleneck_pre_relu', 125, # Appliances
    'mixed4b_3x3_bottleneck_pre_relu', 105, # girders
    'mixed3a_3x3_pre_relu', 54, # swirls
    'head0_bottleneck_pre_relu', 79, # Animal stripes
    'mixed4a_3x3_bottleneck_pre_relu', 2, # x hashing
    'mixed4a_3x3_bottleneck_pre_relu', 14, # windows
    'head0_bottleneck_pre_relu', 84, # Garden ruins
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


# Stage 3a - ring of textures similar to stage 1
for frame in range(start_frame, end_frame):
    fno = frame + 1
    print('Stage 03a, frame {}'.format(fno))

    layer_1 = targets[ 0 ]
    channel_1 = targets[ 1 ]
    layer_2 = targets[ 2 ]
    channel_2 = targets[ 3 ]

    pattern_step = fno % 240

    delta_rot = ( 0.04 ) * ( 0.5 * math.sin( pattern_step / 21.0 ) + math.sin( pattern_step / 35.0 ) )
    delta_zoom = 1.0 - ( 0.0003 ) *  ( 0.5 * math.sin( pattern_step / 15.0 ) + math.sin( pattern_step / 23.0 ) )

    total_rot += delta_rot
    total_zoom *= delta_zoom

    current_img = tfi.affine_zoom( current_img, delta_zoom, delta_rot )
    ref_img = tfi.affine_zoom( img0, total_zoom, total_rot )

    current_img = add_texture_ring( current_img, pattern_step, first_ring_offset_a, ring_stages, layer_1, channel_1 )
    current_img = add_clearing_circle( current_img, pattern_step, second_ring_offset_a, ring_stages, 0.05 )
    current_img = add_clearing_circle( current_img, pattern_step, clear_offset_a, ring_stages, 0.1 )

    current_img = add_texture_ring( current_img, pattern_step, first_ring_offset_b, ring_stages, layer_2, channel_2 )
    current_img = add_clearing_circle( current_img, pattern_step, second_ring_offset_b, ring_stages, 0.05 )
    current_img = add_clearing_circle( current_img, pattern_step, clear_offset_b, ring_stages, 0.1 )

    cropped_img = current_img[margin:-margin, margin:-margin, :]
    tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if (frame % 5 == 0):
        tfi.reset_graph_and_session()

start_frame = 5040
end_frame = 5445

end_colours = np.float32( PIL.Image.open('images/stage03_end_colours.jpeg') )
credit_img = np.float32( PIL.Image.open('images/credits.jpeg') )
complete_fade_img = credit_img  * 0

# Stage 3b - rapid zoom in and fade to credits
for frame in range(start_frame,end_frame):
    fno = frame + 1
    print('Stage 03b - frame {}'.format(fno))

    section_id = ( (fno - start_frame) // 30 ) + 2
    layer_1 = targets[ section_id  * 2 ]
    channel_1 = targets[ section_id * 2 + 1]

    target = tf.reduce_mean( tfi.T(layer_1)[:,:,:,channel_1] )

    delta_rot = 0.1
    delta_zoom = 1.05

    total_rot += delta_rot
    total_zoom *= delta_zoom

    current_img = tfi.affine_zoom( current_img, delta_zoom, delta_rot )

    r = (fno - start_frame)/(end_frame - start_frame)
    mix_amount = 0.99 * (1-r) + 0.96 * r
    current_img = tfi.mix_images( current_img, end_colours, mix_amount )
    current_img = tfi.render_deepdream( target, current_img, iter_n=2, step=1.5, octave_n=4, octave_scale=1.5, direct_objective = True )

    display_img = current_img
    # Fade to black
    if (fno > 5400):
        fade_r = 1.0 - (fno - 5400)/45.0
        display_img = tfi.mix_images( display_img, complete_fade_img, fade_r )

    # Fade in credits
    if (fno > 5430):
        credit_fade = 1.0 - (fno - 5430)/15.0
        display_img = tfi.mix_images( display_img, credit_img, credit_fade )

    cropped_img = display_img[margin:-margin, margin:-margin, :]
    tfi.savejpeg( cropped_img, ('{}/frame_{}.jpeg'.format( directory, '%04d' % fno ) ) )

    if (frame % 5 == 0):
        tfi.reset_graph_and_session()

tfi.close_session()