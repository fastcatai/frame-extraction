# Frame Extraction

This script will extract frames from a bulk of videos.
Additionally, it will detect freezes within a video and outputs the beginning of such a freeze.

## Usage
- Show help: `extract.py -h`, `extract.py videos -h`, `extract.py folder -h`, `extract.py freeze-detection -h`
- Input a list of videos: `extract.py videos /path/to/video1.mkv /path/to/video2.mkv ...`
- Input a folder with videos inside: `extract.py folder /path/to/folder`
- Input a folder with frames to perform only a freeze detection: `extract.py freeze-detection /path/to/folder`
- For defining a output folder just add `--output /path/to/output/folder` to the arguments.
  If no output folder is defined then the parent folder is used.
- To enable freeze detection during extraction add `--freezes` to the arguments.
  Only works for `videos` and `folder` subcommand.
