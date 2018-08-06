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
