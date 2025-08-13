from scipy.optimize import linear_sum_assignment
import numpy as np
import uproot as uproot
from tqdm import tqdm
import pickle

import multiprocessing as mp


#### Common functions
def load_data(filepath, entry_stop=None, load_cms=False):
    with uproot.open(
        filepath,
        num_workers=8,
    ) as f:
        data = {
            "fastsim": {},
        }
        if load_cms:
            data["pflow"] = {}
            data["truth"] = {}
        tree = f[f"evt_tree"]
        for var in tree.keys():
            if not load_cms and ("pflow" in var or "truth" in var):
                continue
            arr = tree[var].array(library="np", entry_stop=entry_stop)
            if "pflow" in var:
                data["pflow"][var.replace("pflow_", "")] = arr
            elif "fastsim" in var:
                data["fastsim"][var.replace("fastsim_", "")] = arr
            elif "truth" in var:
                data["truth"][var.replace("truth_", "")] = arr
            else:
                data[var] = arr
    return data


def zero_neutral_vtx(data):
    for key in ["pflow", "fastsim"]:
        if key not in data:
            continue
        for i in range(len(data[key]["pt"])):
            part_class = data[key]["class"][i]
            data[key]["vx"][i][part_class > 2] = 0
            data[key]["vy"][i][part_class > 2] = 0
            data[key]["vz"][i][part_class > 2] = 0


def get_common_events(data_dict):
    goodEvents = np.array(
        list(
            set.intersection(
                *[set(data["eventNumber"].astype(int)) for data in data_dict.values()]
            )
        ),
        dtype=int,
    )
    return goodEvents


def filter_events(data_dict, n_events=None):
    goodEvents = get_common_events(data_dict)[:n_events]
    for key in data_dict.keys():
        idx_sorted = np.argsort(data_dict[key]["eventNumber"])
        idxs = np.argwhere(
            np.isin(data_dict[key]["eventNumber"][idx_sorted], goodEvents)
        ).flatten()
        for constituent in ["pflow", "fastsim", "truth"]:
            if constituent not in data_dict[key]:
                continue
            for feat_name, value in data_dict[key][constituent].items():
                data_dict[key][constituent][feat_name] = value[idx_sorted][idxs]
        data_dict[key]["eventNumber"] = data_dict[key]["eventNumber"][idx_sorted][idxs]
    print(f"Filtered events: {len(goodEvents)}")


def extract_jet_bkg(data, data_type="jet"):
    new_data = {
        "pflow": {},
        "fastsim": {},
        "truth": {},
    }
    if data_type == "jet":
        mask_fn = lambda x: x > -1
    elif data_type == "bkg":
        mask_fn = lambda x: x == -1

    for key in ["pflow", "fastsim", "truth"]:
        if key not in data:
            continue
        mask = [mask_fn(x) for x in data[key]["idx"]]
        for subkey, value in data[key].items():
            if "jets" in subkey:
                new_data[key][subkey] = value
                continue
            if subkey in ["npflow", "nfastsim", "ntruth"]:
                new_data[key][subkey] = np.array([np.sum(el) for el in mask])
                continue
            new_arr = []
            for i in range(len(value)):
                new_arr.append(value[i][mask[i]])
            new_data[key][subkey] = np.array(new_arr, dtype=object)
    new_data["eventNumber"] = data["eventNumber"]
    return new_data


def reshape_phi(phi):
    if phi.dtype == np.dtype(object):
        return np.array([reshape_phi(el) for el in phi], dtype=object)
    return np.arctan2(np.sin(phi), np.cos(phi))


#### Loading all files
def load_sample(
    path_dict: dict,
    n_events: int = None,
):
    data_dict = {}
    data_dict_all = {}
    data_dict_jet = {}
    data_dict_bkg = {}

    for key, value in path_dict.items():
        key = key.split("_")[0] # Clean name
        print(f"Loading {key} model: {value}")
        data_dict[key] = load_data(
            value, entry_stop=n_events, load_cms=True # if key == "dl" else False
        )
        # zero_neutral_vtx(data_dict[key])
    filter_events(data_dict, n_events=n_events)
    for i, key in enumerate(data_dict.keys()):
        data_dict_jet[key] = extract_jet_bkg(data_dict[key], data_type="jet")
        data_dict_bkg[key] = extract_jet_bkg(data_dict[key], data_type="bkg")
        if "tr" not in data_dict_all and 'truth' in data_dict[key]:
            data_dict_all["tr"] = data_dict[key]["truth"]
            data_dict_jet["tr"] = data_dict_jet[key]["truth"]
            data_dict_bkg["tr"] = data_dict_bkg[key]["truth"]

            data_dict_all["pf"] = data_dict[key]["pflow"]
            data_dict_jet["pf"] = data_dict_jet[key]["pflow"]
            data_dict_bkg["pf"] = data_dict_bkg[key]["pflow"]
        data_dict_all[key] = data_dict[key]["fastsim"]
        data_dict_jet[key] = data_dict_jet[key]["fastsim"]
        data_dict_bkg[key] = data_dict_bkg[key]["fastsim"]

    return data_dict_all, data_dict_jet, data_dict_bkg


def mse(a, b):
    return (a - b) ** 2


def process_event(input_eta, input_phi, truth_eta, truth_phi, dr_cut=0.6):
    truth_eta = np.tile(
        np.expand_dims(truth_eta, axis=1), (1, len(input_eta))
    )  # row content same
    input_eta = np.tile(
        np.expand_dims(input_eta, axis=0), (len(truth_eta), 1)
    )  # column content same

    truth_phi = np.tile(
        np.expand_dims(truth_phi, axis=1), (1, len(input_phi))
    )  # row content same
    input_phi = np.tile(
        np.expand_dims(input_phi, axis=0), (len(truth_phi), 1)
    )  # column content same

    # loss_phi = mse(truth_phi, input_phi)
    # loss_eta = mse(truth_eta, input_eta)
    loss_phi = np.pow(reshape_phi(truth_phi - input_phi), 2)
    loss_eta = np.power(truth_eta - input_eta, 2)

    loss = loss_eta + loss_phi

    loss_hung = loss.copy()

    dr = np.sqrt(loss_eta + loss_phi)
    loss[dr > dr_cut] = 1e3

    loss[loss == np.inf] = 1000
    loss[loss == np.nan] = 1000
    truth_ix, input_ix = linear_sum_assignment(loss)

    # Create boolean mask for dr < 0.6
    mask = dr <= dr_cut

    # Find all pairs (i, j) where dr < 0.6
    filtered_pairs = np.argwhere(mask)

    # Convert truth_ix and input_ix into sets for fast lookups
    truth_input_pairs = set(zip(truth_ix, input_ix))

    # Select new pairs where (i, j) are in the filtered_pairs and truth_input_pairs
    new_pairs = np.array(
        [pair for pair in filtered_pairs if tuple(pair) in truth_input_pairs]
    )

    # Split new pairs into new_truth_ix and new_input_ix
    if len(new_pairs) == 0:
        new_truth_ix = []
        new_input_ix = []
    else:
        new_truth_ix, new_input_ix = new_pairs.T

    input_ix = np.array(new_input_ix)
    truth_ix = np.array(new_truth_ix)

    if len(new_pairs) == 0:
        hung_cost = 1000
        # print(i)
    else:
        indices = input_ix.shape[0] * truth_ix + input_ix
        loss_extract = np.take_along_axis(loss_hung.flatten(), indices, axis=0)
        hung_cost = loss_extract.mean()
    if len(input_ix) > 0:
        assert np.max(input_ix) < input_eta.shape[-1], (
            f"{mask.shape=} {input_ix=} {len(input_eta)=}"
        )
    if len(truth_ix) > 0:
        assert np.max(truth_ix) < truth_eta.shape[0], (
            f"{mask.shape=} {truth_ix=} {len(truth_eta)=}"
        )

    return input_ix, truth_ix, hung_cost


def process_batch(data):
    input_eta, input_phi, truth_eta, truth_phi, dr_cut = data

    input_indices = []
    truth_indices = []

    hung_cost = np.zeros(len(input_eta))

    for i in range(len(input_eta)):
        input_eta_i = input_eta[i]
        input_phi_i = reshape_phi(input_phi[i])

        truth_eta_i = truth_eta[i]
        truth_phi_i = reshape_phi(truth_phi[i])

        input_ix, truth_ix, hung_cost[i] = process_event(
            input_eta_i, input_phi_i, truth_eta_i, truth_phi_i, dr_cut=dr_cut
        )

        input_indices.append(input_ix)
        truth_indices.append(truth_ix)

    return input_indices, truth_indices, hung_cost


def matching_dr(input_data, truth_data, dr_cut=0.6, batch_size=500):
    input_eta = input_data["eta"]
    input_phi = input_data["phi"]

    truth_eta = truth_data["eta"]
    truth_phi = truth_data["phi"]

    input_indices = []
    truth_indices = []
    hung_cost = np.zeros(len(input_eta))

    n_batches = len(input_eta) // batch_size
    n_batches += 1 if len(input_eta) % batch_size != 0 else 0

    batches = np.array_split(np.arange(len(input_eta)), n_batches)
    batched_data = [
        (input_eta[batch], input_phi[batch], truth_eta[batch], truth_phi[batch], dr_cut)
        for batch in batches
    ]

    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(process_batch, batched_data, chunksize=5),
                total=len(batched_data),
            )
        )
    curr_idx = 0
    for input_ix, truth_ix, hung_cost_i in results:
        input_indices.extend(input_ix)
        truth_indices.extend(truth_ix)
        hung_cost[curr_idx : curr_idx + len(hung_cost_i)] = hung_cost_i
        curr_idx += len(hung_cost_i)

    return input_indices, truth_indices, hung_cost


def main():
    # Deep5M + SinCos + DPM++ + EtaEval 2.5 cut
    samples_path_dict = {
        # "JZ3456": {
        #     "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3456_84_25_45k_test_eta25Eval.root"
        # }
        # "JZall": {
        #     "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZall_65_25_154k_test_eta25Eval.root"
        # },
        "JZ7-8": {
            # "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ7-8_65_25_19k_test_eta25Eval.root"
            "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ78_196_25_19k_test_eta25Eval.root"
        },
        # "JZ3-6": {
        #     "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3-6_65_25_45k_test_eta25Eval.root"
        # }
        # "JZ1-2": {
        #     "fm25_path": "/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ1-2_65_25_99k_test_eta25Eval.root"
        # }
    }
    samples_data = {}
    for sample, value in samples_path_dict.items():
        print(f"Loading {sample} sample")
        all_data, jet_data, bkg_data = load_sample(value)
        samples_data[sample] = {"all": all_data, "jet": jet_data, "bkg": bkg_data}

    particle_matching_dicts = {key: {} for key in samples_data.keys()}
    particle_matching_dicts_jet = {key: {} for key in samples_data.keys()}
    particle_matching_dicts_bkg = {key: {} for key in samples_data.keys()}

    print("Matching all particles")
    for sample in samples_data.keys():
        print(f"Processing... {sample} sample")
        for key in samples_data[sample]["all"].keys():
            print(f"Processing... {key} model")
            particle_matching_dicts[sample][key] = matching_dr(
                samples_data[sample]["all"][key],
                samples_data[sample]["all"]["tr"],
                dr_cut=0.6,
            )
    print("Matching jet particles")
    for sample in samples_data.keys():
        print(f"Processing... {sample} sample")
        for key in samples_data[sample]["jet"].keys():
            particle_matching_dicts_jet[sample][key] = matching_dr(
                samples_data[sample]["jet"][key],
                samples_data[sample]["jet"]["tr"],
                dr_cut=0.6,
            )

    print("Matching bkg particles")
    for sample in samples_data.keys():
        print(f"Processing... {sample} sample")
        for key in samples_data[sample]["bkg"].keys():
            particle_matching_dicts_bkg[sample][key] = matching_dr(
                samples_data[sample]["bkg"][key],
                samples_data[sample]["bkg"]["tr"],
                dr_cut=0.6,
            )

    with open(
        # "evals/rcfm_atlas_part_JZ3456_84_25_45k_test_eta25Eval_particle_matching_dicts_all.pkl",
        # "evals/rcfm_atlas_part_JZall_65_25_154k_test_eta25Eval_particle_matching_dicts_all.pkl",
        # "evals/rcfm_atlas_part_JZ7-8_65_25_19k_test_eta25Eval_particle_matching_dicts_all.pkl",
        "evals/rcfm_atlas_part_JZ78_196_25_19k_test_eta25Eval_particle_matching_dicts_all.pkl",
        # "evals/rcfm_atlas_part_JZ1-2_65_25_99k_test_eta25Eval_particle_matching_dicts_all.pkl",
        # "evals/rcfm_atlas_part_JZ3-6_65_25_45k_test_eta25Eval_particle_matching_dicts_all.pkl",
        "wb",
    ) as f:
        pickle.dump(particle_matching_dicts, f)
    with open(
        # "evals/rcfm_atlas_part_JZ3456_84_25_45k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        # "evals/rcfm_atlas_part_JZall_65_25_154k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        # "evals/rcfm_atlas_part_JZ7-8_65_25_19k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        "evals/rcfm_atlas_part_JZ78_196_25_19k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        # "evals/rcfm_atlas_part_JZ1-2_65_25_99k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        # "evals/rcfm_atlas_part_JZ3-6_65_25_45k_test_eta25Eval_particle_matching_dicts_jet.pkl",
        "wb",
    ) as f:
        pickle.dump(particle_matching_dicts_jet, f)
    with open(
        # "evals/rcfm_atlas_part_JZ3456_84_25_45k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        # "evals/rcfm_atlas_part_JZall_65_25_154k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        # "evals/rcfm_atlas_part_JZ7-8_65_25_19k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        "evals/rcfm_atlas_part_JZ78_196_25_19k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        # "evals/rcfm_atlas_part_JZ1-2_65_25_99k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        # "evals/rcfm_atlas_part_JZ3-6_65_25_45k_test_eta25Eval_particle_matching_dicts_bkg.pkl",
        "wb",
    ) as f:
        pickle.dump(particle_matching_dicts_bkg, f)

if __name__ == "__main__":
    main()