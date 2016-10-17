# Python, TensorFlow scripts for Tabea video

This is a set of hacky Python scripts that build frames for a music video I made. This was part
of an end-of-course project at Kadenze called Creative Applications of Deep Learning With TensorFlow.

Course details here: https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i

To use the scripts for anything else than re-building the exact same video, you will need to get into the
code and hack around with hard-coded values and ad-hoc scattered logic. The most complex shared logic
is in the `tfi.py` module, but I have scattered animation choices liberally around the other scripts
to get things done whichever way made sense at the time.

## Pre-requisites

Python 3.5 with numpy, tensorflow, skimage.

To use the video builder shell script, FFMPEG version 2.7 or highers. Or you could use any other
video editor that can work froma collection of still frames.

From http://freemusicarchive.org/music/Ars_Sonor/Raoul_Wallenbergs_Fantastiska_Resa_Genom_Gteborg/05-Tabea

 * Download mp3 file to music/Ars_Sonor_-_05_-_Tabea.mp3

 * Please note this music is not public domain, it is Creative Commons. There is a difference when it comes to sharing or distributing it.
 If you are not sure, just check the license descriptions.

From https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

 * Download inception5h.zip

 * Unzip this file to create `inception` directory containing `tensorflow_inception_graph.pb` file

## Exploring deep dream

There is a script `explore_layers.py` that I used to search for interesting textures - you will
want to edit it to select layers of interest and a start image that you want to see the effects
on. You may want to adjust the Deep Dream parameters. Once done you can run it like this

```bash
python3 explore_layers.py
```

and it will create a directory structure under `explore_layers`

## How to create the animation

### 1. Create start frame

This runs a deep dream on still image. The original still image is taken from a fractal I created a
few years ago.

```bash
python3 make_start_frame.py
```

This creates an additional 1400x840 image in the `images` folder. Note the eventual video has size
1380x720 - the scripts work with a margin of 60 pixels so that corner effects from rotation do not
show up as much.

### 2. Create animation in stages

Run these scripts in order to build different sections of the animation.

```bash
python3 animation_stage_01.py
python3 animation_stage_02.py
python3 animation_stage_03.py
```

Each of these scripts create frame images and other supporting graphics in a directory with matching
name, creating the directory if possible. Between them, they will take up to 2 days to run and render
over 5400 frames.

This last script is an attempt to merge stage 1, which generates successive frames forward (frames 0 to 1200),
with stage 2 which runs backwards, generating frames 4800 to 960 backwards:

```bash
python3 animation_merge_stage.py
```

 . . . it is only partially successful, but I ran out of time to refine this transition further. It
takes about 12 hours to run.

### 3. Build video

This shell script copies the video frames from other parts to a single folder and puts them into
a 30-frames-per-second video, combined with music track, using ffmpeg:

```bash
./make_tabea_video.sh
```

### License

The library file `tfi.py` contains some lines of code by TensorFlow team, which I have modified to fit with
goals of this project. See TENSORFLOW_LICENSE.TXT for Apache 2.0 license which covers original
work by TensorFlow team, seen in that file. Also see LICENSE.TXT for Apache 2.0 license covering
modifications to that file and other code in this project.

Note the license does not cover the images and audio used in the original project. These are licensed
under Creative Commons license https://creativecommons.org/licenses/by/4.0/ should you wish to use them.
Or you could source your own media of course.
