import os
import shutil
from datetime import datetime

import yaml


def prep_out_dir(args, config):
    """
    Copy config file to a new directory - used to store training metadata
    """
    # get a new out dir
    timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
    config["run_name"] = f"{config['name']}_{timestamp}"
    out_dirname = os.path.join("saved_models", config["run_name"])
    if not args.test_run and os.environ.get("LOCAL_RANK") is None:
        os.makedirs(out_dirname, exist_ok=True)

    # copy configs to model save dir
    if not args.test_run and os.environ.get("LOCAL_RANK") is None:
        # get new paths for configs
        new_config_path = os.path.join(out_dirname, os.path.basename(args.config))
        # new_scale_dict_path = os.path.join(
        #     out_dirname, os.path.basename(config["scale_dict"])
        # )
        # new_var_config_path = os.path.join(
        #     out_dirname, os.path.basename(config["var_config"])
        # )

        # copy files and update paths in config object
        # shutil.copyfile(config["scale_dict"], new_scale_dict_path)
        # shutil.copyfile(config["var_config"], new_var_config_path)
        # config["scale_dict"] = new_scale_dict_path
        # config["var_config"] = new_var_config_path

        # dump updated config
        with open(
            os.path.join(out_dirname, os.path.basename(args.config)), "w"
        ) as file:
            yaml.dump(config, file, sort_keys=False)

    return config


def copy_file(in_path, out_path, verbose=False):
    """
    Copys the file from the 'in_path' to the 'out_path',
    including required checks.
    Returns 0 if the file was moved succesfully, and 1 if
    a file already existed in the desired path
    """
    if not os.path.exists(out_path):
        # create enclosing folder
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # copy file and output if required
        if verbose:
            print("Copying file to ", out_path)

        shutil.copyfile(in_path, out_path)
        return 0
    else:
        if verbose:
            print(f"File was already found at {out_path}")
        return 1


def remove_file(path, verbose=False):
    if os.path.exists(path):
        shutil.rmtree(os.path.dirname(path))
        if verbose:
            print(f"Removed training file from temp location: \n\t{path}")
