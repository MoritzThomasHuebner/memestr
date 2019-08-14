import os


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
