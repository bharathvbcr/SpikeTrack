# E7 attribution check: video 47 at 1008 on the e7 binary

Video 47 alone, `--input-size 1008 --max-objects 64`, otherwise the held-out setup, run with the
`speed/e7-geometry` binary (45e5d63, ojas 8b88328; `versions.txt`, `job.sh`). Scored over the 300-frame window
with a one-id split (`47`).

HOTA 0.5149 / DetA 0.4673 / AssA 0.5701 / LocA 0.7238: identical to the committed row in
`../held300-lora600-sperm-detector-max64` (ojas cbd5565, the pre-e7 binary). Boxes and scores differ only in the
last printed float digit. So the 1008 baseline is current, and the E7 drop at 672 comes from the input size.
Mean /frame here: 3.56 (unpaired).
