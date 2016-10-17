# Two copyright notices apply. Some code originally from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
# covered by the following license:

###############################################################################################
#   Copyright 2015, The TensorFlow Authors.
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

# Modifications and additions covered by following license:

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

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
from skimage import transform, draw, filters
import gc
import math

sess = None
model_fn = 'inception/tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    inception_graph_string = f.read()

# This re-loading is necessary to clear out the extra tensors and temp variables we create
# during processing. Otherwise memory use can grow to many GB when processing 1000s of frames
def reset_graph_and_session():
    global sess, t_input

    close_session()
    gc.collect()

    # Define new session, graph and input variable
    sess = tf.Session()
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    graph_def = tf.GraphDef()
    graph_def.ParseFromString( inception_graph_string )
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

def close_session():
    global sess, t_input

    # Clear existing TF data
    if sess:
        sess.close()
    tf.reset_default_graph()

def savejpeg(a, name):
    '''Writes image in Numpy array a to disk in JPEG format'''
    a = np.uint8(np.clip(a/255.0, 0, 1)*255)
    pil_img = PIL.Image.fromarray(a)
    pil_img.save(name, 'jpeg')
    pil_img.close()

def T(layer):
    '''Helper for getting layer output tensor'''
    return sess.graph.get_tensor_by_name("import/%s:0"%layer)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4,
                     octave_scale=1.4, verbose = False, direct_objective = False):
    '''Returns new image derived from img0, that has been changed to increase value of t_obj'''

    if direct_objective:
        t_score = t_obj
    else:
        t_score = tf.reduce_mean(t_obj)

    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = transform.resize(img, np.int32(np.float32(hw)/octave_scale), order=3,
                 clip=False, preserve_range=True).astype(np.float32)
        hi = img - transform.resize(lo, hw, order=3,
                 clip=False, preserve_range=True).astype(np.float32)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if verbose:
            print( '  octave {}'.format( octave ) )
        if octave>0:
            hi = octaves[-octave]
            img = transform.resize(img, hi.shape[:2], order=3,
                 clip=False, preserve_range=True).astype(np.float32) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
    return img

def affine_zoom( img, zoom, spin = 0 ):
    '''Returns new image derived from img, after a central-origin affine transform has been applied'''
    img_copy = img.copy()

    # Shift transforms allow Affine to be applied with centre of image as 0,0
    shift_y, shift_x, _ = (np.array(img_copy.shape)-1) / 2.
    shift_fwd = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    shift_back = transform.SimilarityTransform(translation=[shift_x, shift_y])
    affine = transform.AffineTransform( scale=(zoom, zoom), rotation=(spin * math.pi/180) )

    img_copy = transform.warp( img_copy,
                     ( shift_fwd + ( affine + shift_back )).inverse,
                     order=3,
                     clip=False, preserve_range=True,
                     mode='reflect').astype(np.float32)
    return img_copy

def mix_images( img1, img2, r1 = 0.99 ):
    '''Mixes two images according to fraction desired of first image'''
    return ( img1 * r1 + img2 * (1 - r1 ) )

def circle_mask_blurred( img, radius, sig = 20 ):
    '''Creates a blurred circle, using supplied image as template for dimensions'''
    height, width, ch = img.shape
    centre_y = (height-1) / 2.
    centre_x = (width-1) / 2.
    img_copy = img.copy()
    img_copy[:, :, :] = (0.,0.,0.)
    rr, cc = draw.circle(centre_y, centre_x, radius, img.shape)
    img_copy[rr, cc, :] = (1.,1.,1.)
    img_copy = filters.gaussian(img_copy, sigma=sig, mode='nearest', multichannel=True)
    return img_copy

def ring_mask( img, outer_radius, inner_radius ):
    '''Creates a blurred ring, using supplied image as template for dimensions'''
    return circle_mask_blurred( img, outer_radius ) - circle_mask_blurred( img, inner_radius )

def masked_mix( img_a, img_b, mask_img, mask_mul = 1.0 ):
    '''Combines img_a and img_b using mask_img to control how img_b pixels are mixed'''
    mask = mask_img * mask_mul
    return img_a * (1.0 - mask) + img_b * mask
