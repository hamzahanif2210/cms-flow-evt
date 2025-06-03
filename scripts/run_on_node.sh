source ~/.bashrc
conda activate common
cd /storage/agrp/dreyet/f_delphes/cms-flow-evt

# check if CKPT env variable is set
if [ -z "$CKPT" ]; then
    python train.py -c $CONFIG --gpus 0
else
    echo "Using checkpoint: $CKPT"
    echo
    python train.py -c $CONFIG --gpus 0 --ckpt_path $CKPT
fi
