from .utils import printWithDate

import functools
import sys
import time
import datetime
import os
import os.path
import traceback

import pytz
tz = pytz.timezone('Canada/Eastern')

min_run_time = 60 * 10 # 10 minutes
infos_dir_name = 'runinfos'

class Tee(object):
    #Based on https://stackoverflow.com/a/616686
    def __init__(self, fname, is_err = False):
        self.file = open(fname, 'a')
        self.is_err = is_err
        if is_err:
            self.stdstream = sys.stderr
            sys.stderr = self
        else:
            self.stdstream = sys.stdout
            sys.stdout = self
    def __del__(self):
        if self.is_err:
            sys.stderr = self.stdstream
        else:
            sys.stdout = self.stdstream
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdstream.write(data)
    def flush(self):
        self.file.flush()

def makeLog(logs_prefix, start_time, tstart, is_error, *notes):
    fname = f'error.log' if is_error else f'run.log'
    with open(logs_prefix + fname, 'w') as f:
        f.write(f"start: {start_time.strftime('%Y-%m-%d-%H:%M:%S')}\n")
        f.write(f"stop: {datetime.datetime.now(tz).strftime('%Y-%m-%d-%H:%M:%S')}\n")
        f.write(f"in: {int(tstart > min_run_time)}s\n")
        f.write(f"dir: {os.path.abspath(os.getcwd())}\n")
        f.write(f"{' '.join(sys.argv)}\n")
        f.write('\n'.join([str(n) for n in notes]))

def makelogNamesPrefix(script_name, start_time):
    os.makedirs(infos_dir_name, exist_ok = True)
    os.makedirs(os.path.join(infos_dir_name, script_name), exist_ok = True)
    return os.path.join(infos_dir_name, script_name, f"{start_time.strftime('%Y-%m-%d-%H%M%S-%f')}_")

def logged_main(mainFunc):
    @functools.wraps(mainFunc)
    def wrapped_main(*args, **kwds):
        start_time = datetime.datetime.now(tz)
        script_name = os.path.basename(sys.argv[0])[:-3]
        logs_prefix = makelogNamesPrefix(script_name, start_time)
        tee_out = Tee(logs_prefix + 'stdout.log', is_err = False)
        tee_err = Tee(logs_prefix + 'stderr.log', is_err = True)
        printWithDate(' '.join(sys.argv), colour = 'blue')
        printWithDate(f"Starting {script_name}", colour = 'blue')
        try:
            tstart = time.time()
            val = mainFunc(*args, **kwds)
        except (Exception, KeyboardInterrupt) as e:
            printWithDate(f"Error encountered", colour = 'blue')
            if (time.time() - tstart) > min_run_time:
                makeLog(logs_prefix, start_time, tstart, True, 'Error', e, traceback.format_exc())
            raise
        else:
            printWithDate(f"Run completed", colour = 'blue')
            if (time.time() - tstart) > min_run_time:
                makeLog(logs_prefix, start_time, tstart, False, 'Successful')
        tee_out.flush()
        tee_err.flush()
        return val
    return wrapped_main
