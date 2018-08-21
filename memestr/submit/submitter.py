from collections import OrderedDict
import subprocess
import os
import numpy as np
import time
import uuid


def reformat_input(files, sep=" "):
    """files, a list of files or a string of many files
    each seperated by a space
    """
    if isinstance(files, str):
        files = files.split(" ")
    out = sep.join(files)
    return '"{}"'.format(out)


def command(cmd, sep=" "):
    """cmd: string or list, the command you want to execute,
    examples: "ls -a", ["ls", "-a"]
    sep: string, how you want to split the string into a list
    if the seperator between tokens in the commands is not a space
    """
    if isinstance(cmd, str):
        cmd = cmd.split(sep)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = process.communicate()
    time.sleep(1)
    return out.decode("utf-8").split("\n")


def slurm_handle(piped_input):
    """piped_input: list, this is the output from the command function
    intended for squeue but maybe will be used for other commands
    keep in mind that this modifies the input inplace
    """
    for idx, line in enumerate(piped_input):
        piped_input[idx] = line.split()
    if not piped_input[-1]:
        del piped_input[-1]


def make_bash_script(script_calls, resources, type_script='bash'):
    """iter_items: dict
    resource: SlurmResources, resources for sbatch call
    type_script: string, this will insert the command to
    the program you want to run
    """
    script_call = [type_script]
    format_place_holders = ["{{{}}}".format(key) for key in script_calls.keys()]
    slurm_syntax = "#!/bin/bash\n " \
                   "#SBATCH --ntasks={}\n " \
                   "#SBATCH --mem-per-cpu={}G\n " \
                   "#SBATCH --cpus-per-task {}\n " \
                   "#SBATCH --time={}\n " \
                   "#SBATCH --gres=gpu:{}\n" \
        .format(resources.ntasks, resources.memory, resources.cores, resources.time, resources.ngpu)

    script_command = " ".join(script_call + format_place_holders).format(**script_calls)
    return "\n".join(slurm_syntax + script_command).format(**resources)


class SlurmResources(object):

    def __init__(self, ntasks=1, memory=2, cores=1, hours=24, ngpu=0):
        """

        :param memory: memory in GB
        :param cores: number of cores
        :param hours: time in hours
        :param ngpu: number of gpus
        """
        self.ntasks = ntasks
        self.memory = memory
        self.cores = cores
        self.time_in_hours = hours
        self.ngpu = ngpu

    @property
    def cores(self):
        return self.__cores

    @cores.setter
    def cores(self, cores):
        if not isinstance(cores, int) or isinstance(cores, float):
            raise TypeError("Number of cores must be a number")
        self.__cores = int(cores)

    @property
    def time_in_hours(self):
        return self.__time

    @time_in_hours.setter
    def time_in_hours(self, time_in_hours):
        self.__time = time_in_hours

    @property
    def time(self):
        hours = np.floor(self.__time)
        minutes = np.floor((np.floor(self.__time) - hours) * 60)
        return "{}:{}:00".format(hours, minutes)


class JobSubmitter(object):
    """ initial class object that will take in all of the
    required information to do the work of creating files,
    submitting jobs, etc.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, call_items, sbatch_name=str(uuid.uuid4().hex), submit_dir=dir_path, resources=SlurmResources(),
                 shell_type='bash',
                 iter_dir=None):
        self.call_items = OrderedDict(call_items)
        self.resources = resources
        self.submit_dir = submit_dir
        self.sbatch_name = sbatch_name
        self.shell_type = shell_type
        self.iter_dir = iter_dir

    @property
    def sbatch_name(self):
        return self.__sbatch_name

    @sbatch_name.setter
    def sbatch_name(self, sbatch_name):
        self.__sbatch_name = sbatch_name
        split = sbatch_name.split(".")
        if split[-1] != "sh":
            self.__sbatch_name += ".sh"

    @property
    def sbatch_dir_name(self):
        if self.iter_dir is not None:
            return self.iter_dir
        else:
            split = self.sbatch_name.split(".")
            return '_'.join([i for i in split[:-1]])

    @property
    def submit_dir(self):
        return self.__submit_dir

    @submit_dir.setter
    def submit_dir(self, submit_dir):
        if "~" in submit_dir:
            raise ValueError("~ in place of the home path is not allowed please provide the full path")
        self.__submit_dir = submit_dir

    @property
    def write_dir(self):
        return os.path.join(self.submit_dir, self.sbatch_dir_name)

    @property
    def file_path(self):
        return os.path.join(self.write_dir, self.sbatch_name)

    @property
    def file_written(self):
        return self.file_path

    def _write(self, text):
        if not os.path.isdir(self.write_dir):
            os.makedirs(self.write_dir)
            os.chdir(self.write_dir)
        else:
            raise IOError("{} exists".format(self.write_dir))
        with open(self.file_path, "w") as writing:
            writing.write(text)

    def run(self):
        cmd = make_bash_script(script_calls=self.call_items, resources=self.resources, type_script=self.shell_type)
        file_written = self._write(cmd)
        sbatch_res = command("sbatch {}".format(file_written))
        slurm_handle(sbatch_res)
        return sbatch_res


def find_unallocated_name(name):
    outdir = ''
    for i in range(0, 999):
        outdir = str(i).zfill(3) + "_" + name
        if not os.path.exists(outdir):
            break
    return outdir


def move_log_file_to_outdir(dir_path, outdir, log_file):
    os.rename(dir_path + "/" + log_file, dir_path + "/" + outdir + "/" + log_file)


def run_job(outdir, script, **kwargs):
    script(injection_model=kwargs['injection_model'],
           recovery_model=kwargs['recovery_model'],
           outdir=outdir)
    move_log_file_to_outdir(dir_path=os.path.dirname(os.path.realpath(__file__)),
                            log_file=outdir + '.log',
                            outdir=outdir)
