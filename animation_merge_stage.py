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

start_frame = 960
end_frame = 1190

directory = 'animation_merges'
input_dir1 = 'animation_stage_01'
input_dir2 = 'animation_stage_02'

if not os.path.exists(directory):
    os.makedirs(directory)

# The process works forward then backwards, finding a balance point between two end images
# where the transformations are all self-consistent. In practice, this looks only slightly better
# than a simple fade, and takes many hours to run. However, it *does* actually look better. I think
# a longer overlap period may of looked even better, and/or some different choices of texture to
# use on the overlapped section
merge_end_percents = [100, 98, 96, 92, 84, 68, 0]
fwd_zoom = 0.997
fwd_rot = 0.2
back_zoom = 1.0/0.997
back_rot = -0.2

margin = 60

def target_for_fno( fno ):
    layer = 'mixed5a_3x3_bottleneck_pre_relu'
    channel_1 = 10
    channel_2 = 3
    if ( fno % 240 ) < 120:
        rt = (fno % 240)/120.0
        rit = 1.0 - rt
        target = tf.reduce_mean( rit * tfi.T(layer)[:,:,:,channel_1] + rt * tfi.T(layer)[:,:,:,channel_2] )
    else:
        target = tf.reduce_mean( tfi.T(layer)[:,:,:,channel_2] )
    return target

def fwd_mix_ratio( fno, end_ratio ):
    r = ( (fno - start_frame)/float(end_frame-start_frame) )
    return ( 1 - r ) + r * end_ratio

def back_mix_ratio( fno, end_ratio ):
    r = 1.0 - ( (fno - start_frame)/float(end_frame-start_frame) )
    return ( 1 - r ) + r * end_ratio

def process_image_step( current_img, zoom, rot, mix_ratio, mix_img, target ):
    current_img = tfi.affine_zoom( current_img, zoom, rot )
    current_img = tfi.mix_images( current_img, mix_img, mix_ratio )
    return tfi.render_deepdream( target, current_img, iter_n=2, step=1.5, octave_n=4, octave_scale=1.5, direct_objective = True )

def make_reference_subdir( direction, pct ):
    subdname = '{}/{}_{}'.format( directory, direction, pct )
    if not os.path.exists(subdname):
        os.makedirs(subdname)

def load_reference_img( direction, pct, fno ):
    if pct == 100:
        if direction == 'fwd':
            return np.float32( PIL.Image.open( 'animation_stage_01/overlap_frame_{}.jpeg'.format( '%04d' % fno ) ) )
        else:
            return np.float32( PIL.Image.open( 'animation_stage_02/overlap_frame_{}.jpeg'.format( '%04d' % fno ) ) )
    else:
        subdname = '{}_{}'.format( direction, pct )
        return np.float32( PIL.Image.open( '{}/{}/overlap_frame_{}.jpeg'.format( directory, subdname, '%04d' % fno ) ) )

def save_reference_img( img, direction, pct, fno ):
    subdname = '{}_{}'.format( direction, pct )
    tfi.savejpeg( img, ('{}/{}/overlap_frame_{}.jpeg'.format( directory, subdname, '%04d' % fno ) ) )

def save_rendered_img( img, direction, pct, fno ):
    cropped_img = img[margin:-margin, margin:-margin, :]
    subdname = '{}_{}'.format( direction, pct )
    tfi.savejpeg( cropped_img, ('{}/{}/frame_{}.jpeg'.format( directory, subdname, '%04d' % fno ) ) )

tfi.reset_graph_and_session()
pass_id = 0

for merge_end_id in range( len(merge_end_percents) -1 ):
    pass_id += 1

    prev_end_pct = merge_end_percents[merge_end_id]
    this_end_pct = merge_end_percents[merge_end_id+1]
    end_ratio = this_end_pct/100.0

    # Forward - always start from original reference
    current_img = load_reference_img( 'fwd', 100, start_frame )
    make_reference_subdir( 'fwd', this_end_pct )
    save_reference_img( current_img, 'fwd', this_end_pct, start_frame )
    save_rendered_img( current_img, 'fwd', this_end_pct, start_frame )

    for frame in range(end_frame-start_frame):
        fno = start_frame + frame + 1
        mix_img = load_reference_img( 'back', prev_end_pct, fno )
        mix_ratio = fwd_mix_ratio( fno, end_ratio )
        target = target_for_fno( fno )

        print('Merge 01 & 02, {} pass {}, frame {}, mix ratio {}'.format('fwd', pass_id, fno, mix_ratio))

        current_img = process_image_step( current_img, fwd_zoom, fwd_rot, mix_ratio, mix_img, target )

        save_reference_img( current_img, 'fwd', this_end_pct, fno )
        save_rendered_img( current_img, 'fwd', this_end_pct, fno )

        if (frame % 5 == 0):
            tfi.reset_graph_and_session()

    # Back - always start from original reference
    current_img = load_reference_img( 'back', 100, end_frame + 1 )
    make_reference_subdir( 'back', this_end_pct )
    save_reference_img( current_img, 'back', this_end_pct, end_frame + 1 )
    save_rendered_img( current_img, 'back', this_end_pct, end_frame + 1 )

    for frame in range(end_frame-start_frame):
        fno = end_frame - frame
        mix_img = load_reference_img( 'fwd', prev_end_pct, fno )
        mix_ratio = back_mix_ratio( fno, end_ratio )
        target = target_for_fno( fno )

        print('Merge 01 & 02, {} pass {}, frame {}, mix ratio {}'.format('back', pass_id, fno, mix_ratio))

        current_img = process_image_step( current_img, back_zoom, back_rot, mix_ratio, mix_img, target )

        save_reference_img( current_img, 'back', this_end_pct, fno )
        save_rendered_img( current_img, 'back', this_end_pct, fno )

        if (frame % 5 == 0):
            tfi.reset_graph_and_session()

tfi.close_session()
