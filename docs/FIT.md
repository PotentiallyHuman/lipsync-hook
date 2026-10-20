# Making the added song fit the clip exactly

The clip you upload has no audio. The platform's own library sound is added on top. Sync then depends on one thing:
the library sound must start, under clip time 0, at the exact song time where your hook window started. That number
is in `hook/hook.json` as `cut_start` (and in `DELIVERY.json`). Nothing else: the clip's picture frame 0 is hook
sample 0 by construction (the lead-in is cut on the frame grid).

## The rule
- Pin = `cut_start`, rounded to the editor's grid (most editors show MM:SS:FF at 30 fps: frame = round(cut_start x 30)).
- The clip itself starts at the timeline's 00:00 tick.
- Never adjust the pin by playing the editor and measuring: we measured that loop and it is noise (+13 ms and +98 ms
  for the same pin in two runs; a "correction" from it moved a post 100 ms out of sync).

## Placing it exactly in a timeline editor
1. Drag the sound roughly into place.
2. Zoom the timeline in as far as it goes (ours reached 1600 pixels per second, so 1 px = 0.6 ms).
3. Read two time labels on the ruler and their pixel positions. Pixels per second = (x2 - x1) / (t2 - t1). The target
   pixel for the pin = x1 + (pin - t1) x px/s. If the clip ends before the next label, use two labels below the pin or
   one label plus the minor ticks and extrapolate.
4. Drag the sound's start handle onto that pixel and release. Read the position readout next to the play button: it
   must show the pin. If it does not, correct from the readout, not by ear.
5. Check the clip's left edge sits on the 00:00 tick at that zoom.
6. If the editor shows only the clip's length and the pin is further into the song than the clip is long, trim the
   sound's start in chunks (each at most one visible timeline width), each chunk placed the same way; the chunks
   add up to the pin.

With this, two different clips placed on two different days both landed within 1 ms of their pins.

## The length check before you place
A delivered `clip_silent.mp4` is exactly the hook window long (to one frame). If your uploader compares the clip
length with the hook audio and they differ by more than one frame, the clip is not the one you think: stop.

## What is yours
Uploading, captions, visibility, scheduling and the account are yours to handle. This repo stops at a verified clip
and the number above.
