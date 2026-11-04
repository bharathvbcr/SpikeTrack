#!/bin/bash
# The submission video, bench/results/demo/spiketrack-demo.mp4, built from live runs and the clips
# spiketrack wrote, following docs/demo-script.md. Nothing on screen is mocked up:
# - terminal.txt: the binary's size, linked libraries and --help, captured now;
# - image-36-0/: a live `spiketrack image` run (the script's command) on held-out frame 36/0;
# - ../36-before-after-fp16.mp4: from ../fp16-job.sh (the default as of E30; the cards and narration
#   state the E32 default, whose tracks keep the same id sets, ../../e32-memenc/README.md);
# - title cards: cards.swift (AppKit), narration: macOS `say` (voice Samantha), assembly: ffmpeg.
# Segment specs and narration texts: segments/NN-name.{json,txt}.
# usage: tools/locked.sh demo-video bash bench/results/demo/video/make.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."
V=bench/results/demo/video
B=$V/build
BIN=target/release/spiketrack
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
rm -rf "$B"
mkdir -p "$B"
# DEMO_REUSE_IMAGE=1 keeps the existing image run and binary: only cards, narration and ffmpeg run,
# none of which touches the GPU, so a text-only rebuild can run outside the machine lock.
REUSE=${DEMO_REUSE_IMAGE:-0}
[ "$REUSE" = 1 ] || cargo build --release -j 2 -p spiketrack 2>&1 | tail -1

{
  echo '$ ls -lh target/release/spiketrack'
  ls -lh "$BIN" | awk '{print $1, $5, $NF}'
  echo '$ otool -L target/release/spiketrack | awk '"'"'NR > 1 {print $1}'"'"
  otool -L "$BIN" | awk 'NR > 1 {print "  " $1}'
  echo '$ env -i PATH=/usr/bin:/bin target/release/spiketrack --help | head -4'
  env -i PATH=/usr/bin:/bin "$BIN" --help | head -4
} > "$V/terminal.txt"

[ "$REUSE" = 1 ] && [ -f "$V/image-36-0/overlay.png" ] || {
rm -rf "$V/image-36-0"
"$BIN" image data/visem/VISEM_Tracking_Train_v4/Train/36/images/36_frame_0.jpg \
  --prompt sperm --backend metal --threshold 0.65 --tile 2x2 \
  --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
  --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
  --out "$V/image-36-0" > "$V/image-36-0.log" 2>&1
}
[ -f "$V/image-36-0/overlay.png" ] || { echo "no overlay.png from the image run" >&2; tail -5 "$V/image-36-0.log" >&2; exit 1; }

# Specs: segments/*.json, with "mono_from" expanded into "mono" lines.
python3 -I -c '
import json, pathlib, sys
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for p in sorted(src.glob("*.json")):
    s = json.loads(p.read_text())
    f = s.pop("mono_from", None)
    if f:
        s["mono"] = pathlib.Path(f).read_text().rstrip("\n").split("\n")
    (out / p.name).write_text(json.dumps(s))
' "$V/segments" "$B"
swiftc -O "$V/cards.swift" -o "$B/cards"
"$B/cards" "$B"/*.json

CLIP=bench/results/demo/36-before-after-fp16.mp4
ENC=(-c:v libx264 -pix_fmt yuv420p -r 25 -crf 20 -c:a aac -b:a 160k -ar 48000 -ac 2)
: > "$B/list.txt"
for spec in "$V"/segments/*.json; do
  name=$(basename "$spec" .json)
  say -v Samantha -r 172 -o "$B/$name.aiff" -f "$V/segments/$name.txt"
  speech=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$B/$name.aiff")
  dur=$(python3 -I -c "print(round($speech + 1.2, 2))")
  # 0.4 s of silence before the voice and 0.8 s after it.
  audio="[1:a]adelay=400|400,apad=whole_dur=$dur,aformat=sample_rates=48000:channel_layouts=stereo[a]"
  if [ "$name" = 05-track ]; then
    ffmpeg -hide_banner -loglevel error -y -loop 1 -framerate 25 -i "$B/$name.png" -i "$B/$name.aiff" \
      -stream_loop -1 -i "$CLIP" \
      -filter_complex "[0:v][2:v]overlay=0:96:shortest=0[v];$audio" -map "[v]" -map "[a]" -t "$dur" \
      "${ENC[@]}" "$B/$name.mp4"
  else
    ffmpeg -hide_banner -loglevel error -y -loop 1 -framerate 25 -i "$B/$name.png" -i "$B/$name.aiff" \
      -filter_complex "$audio" -map 0:v -map "[a]" -t "$dur" "${ENC[@]}" "$B/$name.mp4"
  fi
  echo "file '$name.mp4'" >> "$B/list.txt"
  echo "$name: ${dur} s"
done
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$B/list.txt" -c copy bench/results/demo/spiketrack-demo.mp4
ffprobe -v error -show_entries format=duration,size -of default=nw=1 bench/results/demo/spiketrack-demo.mp4
