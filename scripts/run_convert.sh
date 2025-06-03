source ~/.bashrc
conda activate common
cd /storage/agrp/dreyet/f_delphes/cms-flow-evt/

python utils/convert_full_event.py -i ${INFILE} -o ${OUTFILE} -n 500 -maxN 100000