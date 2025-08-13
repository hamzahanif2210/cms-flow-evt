# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ3456_84_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ3-6.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3456_84_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZall_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ1-8.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZall_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ1-2_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ1-2.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ1-2_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ3-6_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ3-6.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3-6_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ7-8_65_25.root"
eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ78_196_25.root"
test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ7-8.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ7-8_65_25"
out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ78_196_25"

python evaluation/preprocess_eval_fast.py \
  -e ${eval_file} \
  -d ${test_file} \
  -o ${out_file} \
  -n -1 \
  -dr 0.4 \
  --eta 2.5