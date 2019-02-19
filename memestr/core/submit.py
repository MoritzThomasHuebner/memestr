import os


def find_unallocated_name(name):
    items = next(os.walk('.'))
    outdirs = items[1]
    for i in range(0, 999):
        for outdir in outdirs:
            string = str(i).zfill(3) + "_" + name
            if string in outdir:
                break
            elif outdir == outdirs[-1]:
                return str(i).zfill(3) + "_" + name
    raise RuntimeError('Not able to find an unused out directory name')


def get_injection_parameter_set(id):
    injection_params = {}
    with open('parameter_sets/' + str(id)) as f:
        complete_file = f.read()
        attributes = complete_file.split(' ')
        for attribute in attributes:
            if attribute:
                key_value = attribute.split('=')
                if len(key_value) > 1:
                    injection_params[key_value[0]] = float(key_value[1])
    return injection_params


def get_injection_bash_strings(id):
    params = get_injection_parameter_set(id)
    res = ''
    for param, val in params.items():
        res = res + param + '=' + str(val) + ' '
    return res


def create_injection_parameter_set(size, sampling_function):
    if not os.path.isdir('parameter_sets'):
        os.mkdir('parameter_sets')
    for id in range(size):
        parameters = sampling_function()
        with open('parameter_sets/' + str(id), 'w') as f:
            for key, value in parameters.items():
                f.write(key + '=' + str(value) + ' ')
            f.write('\n')


def move_log_file_to_outdir(dir_path, outdir, log_file):
    os.rename(dir_path + "/" + log_file, dir_path + "/" + outdir + "/" + log_file)


def run_job(outdir, script, **kwargs):
    result = script(outdir=outdir, **kwargs)
    return result


def parse_kwargs(input):
    kwargs = dict()
    for arg in input:
        key = arg.split("=")[0]
        value = arg.split("=")[1]
        if any(char.isdigit() for char in value):
            if all(char.isdigit() for char in value):
                value = int(value)
            else:
                value = float(value)
        kwargs[key] = value
    return kwargs