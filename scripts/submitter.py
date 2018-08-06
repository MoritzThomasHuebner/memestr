from collections import OrderedDict
import subprocess
import os
import numpy as np

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


def make_bash_script(iter_items, resource, type_script='bash'):
    """iter_items: dict
    resource: dict, resources for sbatch call
    type_script: string, this will insert the command to
    the program you want to run
    """
    script_call = [type_script]
    format_place_holders = ["{{{}}}".format(key) for key in iter_items.keys()]
    cmdstr = [
        "#!/bin/bash\n", "#SBATCH --mem={mem}G", "#SBATCH -c {cores}",
        "#SBATCH --time={time}", "#SBATCH --gres=gpu:{ngpu}\n"
    ]
    for key in ["mem", "cores", "time", "ngpu"]:
        if key not in resource.keys():
            raise ValueError(
                "resource dictionary should have the key: {}".format(key)
            )
    scmd = " ".join(script_call + format_place_holders).format(**iter_items)
    return "\n".join(cmdstr + [scmd]).format(**resource)


class SlurmResources(object):

    def __init__(self, memory=2, cores=1, hours=24, ngpu=0):
        """

        :param memory: memory in GB
        :param cores: number of cores
        :param hours: time in hours
        :param ngpu: number of gpus
        """
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
        days = np.floor(self.__time/24)
        tmp = self.__time % 24
        hours = np.floor(self.__time/24)
        tmp = self.__time - hours
        minutes =


class JobSubmitter(object):
    """ initial class object that will take in all of the
    required information to do the work of creating files,
    submitting jobs, etc.
    """
    def __init__(self, call_items, resources, submit_dir, sbatch_name, prog_type=None, iter_dir=None):
        self.call_items = OrderedDict(call_items)
        self.resources = OrderedDict(resources)
        self.submit_dir = submit_dir
        self.sbatch_name = sbatch_name
        self.prog_type = prog_type
        self.iter_dir = iter_dir

    def _write(self, text):
        split_name = self.sbatch_name.split(".")
        if split_name[-1] != "sh":
            raise Exception(
                "make sure that your file ends with .sh (e.g test.sh)"
            )
        if self.iter_dir is not None:
            sbatch_dir_name = self.iter_dir
        else:
            sbatch_dir_name = '_'.join([i for i in split_name[:-1]])
        if "~" in self.submit_dir:
            raise Exception(
                "~ in place of the home path is not allowed please provide the full path"
            )
        write_dir = os.path.join(self.submit_dir, sbatch_dir_name)
        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)
            os.chdir(write_dir)
        else:
            raise FileExistsError(
                "{} exists".format(write_dir)
            )
        file_path = os.path.join(write_dir, self.sbatch_name)
        with open(file_path, "w") as writing:
            writing.write(text)
        self.file_written = file_path
        return file_path

    def run(self):
        cmd = make_call_cmd(self.call_items, self.resources, self.prog_type)
        file_written = self._write(cmd)
        sbatch_res = command("sbatch {}".format(file_written))
        slurm_handle(sbatch_res)
        return sbatch_res