import os
import sys

ncpus = "6"
ngpus = "1"
mem = "30gb"
io = "5"
gputype = "A6000"
config = "configs/evt_atlas.yaml"
ckpt = "/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_net_JZall_20250430-T153307/ckpts/fm_atlas_pow_evt_net_JZall_20250430-T153307-epoch=400-val_loss_avg=0.5722.ckpt"

# config = "configs/part_atlas.yaml"
# ckpt = "/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZall_20250430-T153058/ckpts/rcfm_atlas_part_JZall_20250430-T153058-epoch=65-val_loss_avg=1.7316.ckpt"

if len(sys.argv) > 1:
    run_eval = sys.argv[1] == "eval"
else:
    run_eval = False

if run_eval:
    walltime = "32:00:00"
else:
    walltime = "72:00:00"

command = f"qsub -o {os.getcwd()}/saved_models/output.log"
command += f" -e {os.getcwd()}/saved_models/error.log"
command += f" -q N -N flow_atlas_fevt -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io={io},gputype={gputype}"
command += f" -v CONFIG={config}"
if ckpt is not None:
    command += f",CKPT={ckpt}"

if run_eval:
    command += f" {os.getcwd()}/eval_on_node_fevt.sh"
else:
    command += f" {os.getcwd()}/run_on_node.sh"

print(command)
os.system(command)
