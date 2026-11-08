#!/usr/bin/env bash
# Download the single participant used in the notebook (+ stimulus annotations) from the OpenNeuro S3 bucket.
set -e
SUB=${1:-sub-20}
S3=https://s3.amazonaws.com/openneuro.org/ds003688
OUT=data/ds003688
P=$SUB/ses-iemu/ieeg
mkdir -p $OUT/$P $OUT/stimuli/annotations/sound
for f in participants.tsv participants.json dataset_description.json; do curl -sfL $S3/$f -o $OUT/$f; done
for f in ${SUB}_ses-iemu_task-film_acq-clinical_run-1_ieeg.{vhdr,vmrk,eeg,json} \
         ${SUB}_ses-iemu_task-film_acq-clinical_run-1_channels.tsv ${SUB}_ses-iemu_task-film_run-1_events.tsv \
         ${SUB}_ses-iemu_acq-clinical_electrodes.tsv ${SUB}_ses-iemu_acq-clinical_coordsystem.json; do
  echo "$f"; curl -sfL $S3/$P/$f -o $OUT/$P/$f
done
for f in sentences words phonemes syllables clauses; do
  curl -sfL $S3/stimuli/annotations/sound/sound_annotation_$f.tsv -o $OUT/stimuli/annotations/sound/sound_annotation_$f.tsv
done
echo "done → $OUT/$P"
