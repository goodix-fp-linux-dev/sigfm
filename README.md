# SIGFM

SIGFM stands for "SIFT Is Good For Matching"

This is a new fingerprint matcher algorithm designed for low resolution sensors.

This algo aims to replace [libfprint](https://gitlab.freedesktop.org/libfprint)'s default matching algorithm.

It uses SIFT at its core.

SIGFM is meant to work with 64x80 images but it may works with other resolutions.

To crop 5110's image down to meaningful part (i.e. 64x80), use the following command: ```mogrify -crop 64x80+0+0 -format jpg ./fingerprint.pgm``` .
Note that you need ```imagemagick``` installed for this to work