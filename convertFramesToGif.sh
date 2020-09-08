#! /bin/bash
ffmpeg -f image2 -framerate 12 -start_number 0988800 -i ./graphs/animFrames/120%07d.jpg animGraph.gif
