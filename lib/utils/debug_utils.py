import os
from termcolor import colored
import subprocess
import time
import shutil


def toc():
    return time.time() * 1000

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def log_time(text):
    strf = get_time()
    print(colored(strf, 'yellow'), colored(text, 'green'))

def mywarn(text):
    myprint(text, 'warn')

warning_infos = set()

def oncewarn(text):
    if text in warning_infos:
        return
    warning_infos.add(text)
    myprint(text, 'warn')


def myerror(text):
    myprint(text, 'error')

def run_cmd(cmd, verbo=True, bg=False):
    if verbo: myprint('[run] ' + cmd, 'run')
    if bg:
        args = cmd.split()
        print(args)
        p = subprocess.Popen(args)
        return [p]
    else:
        exit_status = os.system(cmd)
        if exit_status != 0:
            raise RuntimeError
        return []

def mkdir(path):
    if os.path.exists(path):
        return 0
    log('mkdir {}'.format(path))
    os.makedirs(path, exist_ok=True)

def cp(srcname, dstname):
    mkdir(os.join(os.path.dirname(dstname)))
    shutil.copyfile(srcname, dstname)

def check_exists(path):
    flag1 = os.path.isfile(path) and os.path.exists(path)
    flag2 = os.path.isdir(path) and len(os.listdir(path)) >= 10
    return flag1 or flag2