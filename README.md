# Decoding speech vs. non-speech from intracranial EEG (ES 673 – Assignment 5)

Decodes whether a 1-s window of iEEG activity was recorded while the participant heard **speech** or
**music** (non-speech) in a short film. Data: OpenNeuro **ds003688** (v1.0.7), participant `sub-20`,
`ses-iemu`, `task-film`, `acq-clinical`, `run-1`.

## Files

| file | purpose |
|---|---|
| `speech_decoder.ipynb` | the executed notebook (code + outputs); submit this |
| `speech_decoder.py` | same notebook as a plain script (jupytext "percent" format); `python speech_decoder.py` runs everything |
| `report.md` / `report.pdf` | the analysis report; `python build_report.py` rebuilds the PDF from the markdown (uses LibreOffice) |
| `report.tex` | the same report as LaTeX, for editing in Overleaf (`pdflatex report.tex` twice) |
| `download_data.sh` | fetches the one participant used (~90 MB) plus the stimulus annotations from OpenNeuro S3 |
| `figures/`, `results.csv`, `results.json`, `electrode_indices.csv` | outputs written by the notebook |

## Run

```bash
pip install -r requirements.txt
bash download_data.sh            # → data/ds003688/...
python speech_decoder.py         # ~1–2 min with a GPU, writes figures/ and results.*
# or: jupytext --to ipynb speech_decoder.py && jupyter nbconvert --to notebook --execute --inplace speech_decoder.ipynb
python build_report.py           # report.md → report.pdf (needs LibreOffice)
# or: pdflatex report.tex && pdflatex report.tex   # same report from the LaTeX source
```

## Pipeline (one notebook, top to bottom)

1. **Load** the BrainVision recording with `mne.io.read_raw_brainvision`; channel types / bad channels from `channels.tsv`.
2. **Preprocess**: notch 50/100/150/200 Hz, common-average reference over the good ECoG channels.
3. **Features**: log band-power envelopes (delta … high-frequency band 70–170 Hz) via band-pass + Hilbert, binned at 16 Hz.
4. **Align** with the speech / music blocks in `events.tsv`; 1-s windows, 0.5-s stride, each tagged with its block.
5. **Decoders**: logistic regression on band power, 1-D CNN on the raw signal, LSTM on the band-power sequence;
   **leave-one-block-out** cross-validation (random window splits leak because neighbouring windows overlap).
6. **Evaluation**: accuracy, balanced accuracy, F1, ROC-AUC, confusion matrices, per-block accuracy, decoded-probability timeline.
7. **Explorations**: random vs. block-wise split, single frequency bands, window length.
8. **Extra credit**: per-electrode speech selectivity (speech vs. music) and acoustic-onset response (sentence onsets) mapped on the electrode coordinates.
