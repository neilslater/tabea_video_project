cp animation_stage_01/frame_*.jpeg tabea_video/
cp animation_stage_02/frame_*.jpeg tabea_video/
cp animation_merges/fwd_0/frame_*.jpeg tabea_video/
cp animation_stage_03/frame_*.jpeg tabea_video/
ffmpeg -y -framerate 30 -start_number 0 -i tabea_video/frame_%04d.jpeg -i music/Ars_Sonor_-_05_-_Tabea.mp3 -c:v libx264 -c:a libfaac -r 30 -pix_fmt yuv420p -vb 10M -ab 128k tabea_video.mp4
