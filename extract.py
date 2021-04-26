import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def input_video_list(videos: list[Path], freeze_detection: bool, output_folder: Path = None):
    """Is called when a list of video paths are inputted as arguments"""

    # check if video paths are files and exists
    for video in videos:
        if not video.is_file():
            print('\n[ERROR]', video, 'does not exists or is not a file\n', file=sys.stderr)
            return
    # print message
    video_count = len(videos)
    video_str = 'video' if video_count == 1 else 'videos'
    print('Processing {} {}: Depending on the number of videos this could take some time.'
          .format(video_count, video_str))
    # go through each video
    for video in videos:
        # reset states
        reset()
        # extract
        print('\nVideo:', video)
        frame_extraction(video, freeze_detection, output_folder)


def input_folder(folder: Path, freeze_detection: bool, output_folder: Path = None):
    """Is called when a folder with videos is inputted as argument"""

    # check if path is directory
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    # get all files of folder
    files = [p for p in folder.glob('*') if p.is_file()]
    # print message
    file_count = len(files)
    file_str = 'file' if file_count == 1 else 'files'
    print('Processing {} {}: Depending on the number of videos this could take some time. Non-video files are skipped.'
          .format(file_count, file_str))
    # go through each file in folder
    for file in files:
        # reset states
        reset()
        # extract
        print('\nFile:', file)
        frame_extraction(file, freeze_detection, output_folder)


def input_frame_folder(folder: Path, output_folder: Path = None):
    """Freeze detection on a folder that contains only video frames"""

    # reset states
    reset()
    # check if path is directory
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    # get all files of folder
    files = [p for p in folder.glob('*') if p.is_file()]
    files = sorted(files, key=lambda x: int(x.with_suffix('').name))
    # freeze detection
    freeze_frames = []
    for file in tqdm(files):
        frame = cv2.imread(file.__str__())
        freeze_begins = perform_freeze_detection(frame)
        if freeze_begins:
            freeze_frames.append(int(file.with_suffix('').name))
    # write file with freeze frames
    if output_folder is None:
        output_folder = folder.parent
    freezes_file = output_folder.joinpath(folder.name + FREEZES_EXT)
    write_freeze_frames(freeze_frames, freezes_file)


def frame_extraction(video_path: Path, freeze_detection: bool, output_folder: Path = None):
    """Creates a folder within 'output_folder' with the name of the video and extracts and saves all frames from
    video. Additionally saves a file within 'output_folder' with a list of freeze frames, if 'freeze_detection' is
    enabled.
    """

    # open video file
    cap = cv2.VideoCapture()
    video_file = video_path.__str__()
    is_open = cap.open(video_file)
    if not is_open:
        print('\n[INFO] Skip {} because it is not a video or could not be opened'.format(video_file))
        return

    # get video name without extension and create frame folder if not existent
    if output_folder is None:
        output_folder = video_path.parent
    video_filename = video_path.with_suffix('').name
    frame_folder = output_folder.joinpath(video_filename)
    try:
        frame_folder.mkdir()
    except FileExistsError:
        print('\n[ERROR] {} already exists\n'.format(frame_folder), file=sys.stderr)
        return

    # get total frames count of video
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get frames an save them
    progressbar = tqdm(total=frame_count)
    freeze_frames = []
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # save frame
        filename = '{}/{}.png'.format(frame_folder.__str__(), count)
        cv2.imwrite(filename, frame)

        # perform freeze detection if flag is set
        if freeze_detection:
            freeze_begins = perform_freeze_detection(frame)
            if freeze_begins:
                freeze_frames.append(count)

        count += 1
        progressbar.update()
    cap.release()
    progressbar.close()

    # write file with freeze frames
    if freeze_detection:
        freezes_file = output_folder.joinpath(video_filename + FREEZES_EXT)
        write_freeze_frames(freeze_frames, freezes_file)


def write_freeze_frames(freeze_frames: list[int], output_file: Path):
    # if file exits then create another one with timestamp
    if output_file.exists():
        from datetime import datetime
        filename = output_file.with_suffix('').name
        output_file = output_file.parent.joinpath(
            '{}({}){}'.format(filename, datetime.today().strftime('%Y-%m-%dT%H-%M-%S'), FREEZES_EXT))
    # generate content
    content = '\n'.join(map(lambda n: str(n), freeze_frames))
    with open(file=output_file, mode='w') as f:
        f.write(content)


def perform_freeze_detection(frame: np.ndarray):
    """Freeze detection by taking the difference between the current frame and
    the previous frame. It then look for a freeze within a specified window size.
    """
    global prev_frame, detected

    # calc the difference (array with zeros if it is the first frame)
    diff = cv2.absdiff(frame, np.zeros(frame.shape, dtype=np.uint8) if prev_frame is None else prev_frame)
    # total number of pixels
    pixels = np.array([frame.shape[0] * frame.shape[1]])
    # convert to HSV color space
    diff_hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    # we only need S-/V-values and calc average
    summed_diff = cv2.sumElems(diff_hsv)[1:3]
    avg_sv = summed_diff / pixels
    avg = np.sqrt(np.sum(np.power(avg_sv, 2)))
    # add average to list to calc a window average later
    intensity_avg.append(avg)
    # flag if at this frame a freeze begins
    start_freeze = False
    # if previous frame is available and we have enough values, then
    # take a window average and look if it reaches a defined threshold
    # that indicated if a freeze happened within that window
    if prev_frame is not None and len(intensity_avg) >= window_size:
        win_avg = sum(intensity_avg) / len(intensity_avg)
        # freeze detected
        if win_avg <= 50 and not detected:
            start_freeze = True
            detected = True
        # freeze phase ended
        if win_avg > 75 and detected:
            detected = False
        # drop oldest value since it
        # is not needed in the future
        intensity_avg.pop(0)
    # save frame for next frames
    prev_frame = frame
    return start_freeze


def reset():
    global intensity_avg, prev_frame, detected
    # reset
    intensity_avg = []
    prev_frame = None
    detected = False


FREEZES_EXT = '.freezes.txt'
# number of frames that are
# averaged to look for a freeze
window_size = 10
# list of average intensity values
# usually the same size as windows_size
# because the oldest value is removed
intensity_avg = []
# holds the previous frame
# for calc the difference
prev_frame = None
# flag that indicated if we
# are within a freeze phase
detected = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame extraction and freeze detection for a bulk of videos')
    input_type = parser.add_subparsers(dest='input_type', help='available input methods')

    video_parser = input_type.add_parser('videos', help='list of video paths with at least one video')
    video_parser.add_argument('-f', '--freezes', action='store_true', help='enables freeze detection if set')
    video_parser.add_argument('videos', metavar='VIDEO', type=Path, nargs='+', help='list of video paths')

    folder_parser = input_type.add_parser('folder', help='folder containing video files (non-video files are skipped)')
    folder_parser.add_argument('-f', '--freezes', action='store_true', help='enables freeze detection if set')
    folder_parser.add_argument('folder', metavar='FOLDER', type=Path, help='folder containing only video files')

    freeze_parser = input_type.add_parser('freeze-detection', help='folder containing only video frames')
    freeze_parser.add_argument('folder', metavar='FRAME-FOLDER', type=Path, help='folder containing only video frames')

    parser.add_argument('-o', '--output', type=Path, help='folder for storing frames of all videos, if empty then '
                                                          'frame folder will be created in parent directory of video')
    args = parser.parse_args()

    # check if output is folder
    if args.output is not None and not args.output.is_dir():
        print('\n[ERROR] {} is not a directory or does not exists\n'.format(args.output), file=sys.stderr)
    else:
        if args.input_type == 'videos':
            input_video_list(args.videos, args.freezes, args.output, )
        elif args.input_type == 'folder':
            input_folder(args.folder, args.freezes, args.output)
        elif args.input_type == 'freeze-detection':
            input_frame_folder(args.folder, args.output)
