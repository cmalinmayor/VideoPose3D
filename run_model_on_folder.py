import os
import argparse
import subprocess
import sys
from detectron2.utils.logger import setup_logger
from inference import infer_video_d2

DATASET_DIR = ''
DATASET_NAME = ''






def list_videos(path_to_folder):
    return [f for f in os.listdir(path_to_folder) if f.endswith('.mp4')]


def infer_videos_2d(path_to_input_folder, path_to_output_folder):
    print('Marking videos')
    setup_logger()
    namespace = argparse.Namespace(cfg='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
                                   output_dir=path_to_output_folder,
                                   image_ext='mp4',
                                   im_or_folder=path_to_input_folder)
    print(namespace)


    infer_video_d2.main(namespace)


def create_custom_dataframe(path_for_marked_videos, dataset_name, parent_folder):
    print('creating dataframe')
    DATASET_DIR = parent_folder + r'\custom_dataframes'
    DATASET_NAME = dataset_name
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    args = [sys.executable,
            "prepare_data_2d_custom.py",
            '-i',
            path_for_marked_videos,
            '-o',
            DATASET_NAME
    ]
    subprocess.run(args ,cwd=parent_folder +"/data/")


def process_videos( videos_folder, output_folder, dataset_name):
    files = list_videos(videos_folder)
    for file in files:
        args = [sys.executable,
                "run.py",
                '-d',
                'custom',
                '-k',
                dataset_name,
                '-arc',
                '3,3,3,3,3',
                '-c',
                'checkpoint',
                '--evaluate',
                'pretrained_h36m_detectron_coco.bin',
                '--render',
                '--viz-subject',
                file,
                '--viz-action',
                'custom',
                '--viz-camera',
                '0',
                '--viz-video',
                videos_folder + '/' + file,
                '--viz-output',
                output_folder + '/output_' + file,
                '--viz-size',
                '6'
                ]
        subprocess.run(args)


if __name__ == '__main__':
    parent_folder = str(os.getcwd())
    print(parent_folder)
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('videos_folder', type=str,
                        help='A folder with videos to process')
    parser.add_argument('marked_folder', type=str,
                        help='A folder with processed videos')
    parser.add_argument('results_folder', type=str,
                        help='A required integer positional argument')
    parser.add_argument('-dataset_name', type=str,
                        help='Custom name for dataset')

    args = parser.parse_args()

    infer_videos_2d(args.videos_folder, args.marked_folder)

    if args.dataset_name is not None:
        create_custom_dataframe(args.marked_folder, args.dataset_name, parent_folder)
        process_videos( args.videos_folder, args.results_folder, args.dataset_name)
    else:
        dataset_name = args.marked_folder.replace(r'\\', '/').replace('.', '').split('/')[-1]
        create_custom_dataframe(args.marked_folder, dataset_name, parent_folder)
        process_videos(args.videos_folder, args.results_folder, dataset_name)


