#! /bin/bash
ffmpeg -f image2 -framerate 12 -i ./graphs/animFrames/animGraph_%04d.jpg animGraph.gif
