import os
import shutil


def recreate_dir(path_dir):
    if os.path.exists(path_dir):
        shutil.rmtree(path_dir)
    os.makedirs(path_dir)


def mkdir_if_not_exists(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def rm_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
