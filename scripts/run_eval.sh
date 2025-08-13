
# part_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZ3456_20250427-T130542/part_atlas.yaml"
# part_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZ3456_20250427-T130542/ckpts/rcfm_atlas_part_JZ3456_20250427-T130542-epoch=84-val_loss_avg=0.9819.ckpt"
# part_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZall_20250430-T153058/part_atlas.yaml"
# part_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZall_20250430-T153058/ckpts/rcfm_atlas_part_JZall_20250430-T153058-epoch=65-val_loss_avg=1.7316.ckpt"
part_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZ78_20250513-T144154/part_atlas.yaml"
part_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_JZ78_20250513-T144154/ckpts/rcfm_atlas_part_JZ78_20250513-T144154-epoch=196-val_loss_avg=0.8804.ckpt"

# evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_cms_pow_evt_net_JZ3456_20250423-T103135/evt.yaml"
# evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_cms_pow_evt_net_JZ3456_20250423-T103135/ckpts/fm_cms_pow_evt_net_JZ3456_20250423-T103135-epoch=497-val_loss_avg=0.9513.ckpt"
# evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_net_JZall_20250430-T153307/evt_atlas.yaml"
# evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_net_JZall_20250430-T153307/ckpts/fm_atlas_pow_evt_net_JZall_20250430-T153307-epoch=400-val_loss_avg=0.5722.ckpt"
evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_net_JZ78_20250513-T150242/evt_atlas.yaml"
evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_net_JZ78_20250513-T150242/ckpts/fm_atlas_pow_evt_net_JZ78_20250513-T150242-epoch=445-val_loss_avg=1.3999.ckpt"

# test_path="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ1-2.root"
# test_path="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ3-6.root"
test_path="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ7-8.root"

python eval.py \
  --config ${part_config} \
  --checkpoint ${part_ckpt} \
  --config_evt ${evt_config} \
  --checkpoint_evt ${evt_ckpt} \
  --n_steps 25 \
  --gpu 0 \
  --num_events -1 \
  --batch_size 1000 \
  --test_path ${test_path}
  # --prefix "JZ3-6"
#   --eval_dir EVAL_DIR \