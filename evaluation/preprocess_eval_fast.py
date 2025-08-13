### From https://github.com/dkobylianskii/f_delphes/blob/dkobylia-fevt/evaluation/preprocess_eval_fast.py

import argparse

import numpy as np
import fastjet as fj
import awkward as ak

from tqdm import tqdm
from itertools import product

import multiprocessing as mp

import uproot
import sys, os
from contextlib import contextmanager

from jet_helper import Jet, get_cluster_sequence

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


# coord_var_list = ["px", "py", "pz", "pt", "eta", "phi", "mass"]
# vertex_var_list = ["vx", "vy", "vz"]

# var_dict = {
#     "events": ["event"],
#     "gens": coord_var_list + ["pdgId"] + vertex_var_list,
#     "pfcs": coord_var_list + ["pdgId"] + vertex_var_list,
# }

# prefix_dict = {
#     "gens": "GenPart_",
#     "pfcs": "PFCand_",
# }

varlist = ["pt", "eta", "phi", "class"]


parser = argparse.ArgumentParser(description="Preprocess full event files")
parser.add_argument("-j", "--jet", type=str, required=False, help="Input jet file")
parser.add_argument("-b", "--bkg", type=str, required=False, help="Input bkg file")
parser.add_argument("-e", "--evt", type=str, required=False, help="Input evt file")
parser.add_argument(
    "-d", "--data", type=str, required=True, help="Input test data file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=100
)
parser.add_argument("-dr", type=float, help="Delta R for jet clustering", default=0.5)
parser.add_argument("-no_clustering", action="store_true", help="Skip jet clustering")
parser.add_argument("-dl", action="store_true", help="Treat as delphes file")
parser.add_argument("-df", action="store_true", help="Treat as diffusion file")
parser.add_argument("--eta", type=float, help="Eta cut", default=3.0)


def load_file(filename, ttype="jet", entry_start=0, n_events=None, fs=False, df=False):
    truth_data = {var: None for var in varlist}
    pflow_data = {var: None for var in varlist}
    with uproot.open(filename, num_workers=4) as file:
        tree = file[f"{ttype}_tree"]
        if n_events is None:
            n_events = tree.num_entries
        varlist_ = varlist if not fs else varlist + ["ind"]
        for var in varlist_:
            truth_data[var] = tree[f"truth_{var}"].array(
                library="np",
                entry_stop=n_events + entry_start,
                entry_start=entry_start,
            )
            if df:
                pflow_tree = f"fastsim_{var}"
            else:
                pflow_tree = f"pflow_{var}"
            pflow_data[var] = tree[pflow_tree].array(
                library="np",
                entry_stop=n_events + entry_start,
                entry_start=entry_start,
            )
        event_number = tree["eventNumber"].array(
            library="np",
            entry_stop=n_events + entry_start,
            entry_start=entry_start,
        )
        truth_data["eventNumber"] = event_number
        pflow_data["eventNumber"] = event_number
    return truth_data, pflow_data, event_number


def to_ak(pt, eta, phi):
    return ak.Array(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
            "E": pt * np.cosh(eta),
        },
        with_name="Momentum4D",
    )


def find_repeats(arr):
    return {el: np.argwhere(el == arr).flatten() for el in np.unique(arr)}


def cluster_jets(pt, eta, phi, jetdef, ptmin=20):
    particles = to_ak(pt, eta, phi)
    cs = get_cluster_sequence(
        jetdef, particles, user_indices=list(range(len(particles)))
    )
    jets = cs.inclusive_jets(ptmin=ptmin)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, 0.5, calc_substructure=True) for j in jets]
    jets = [j for j in jets if j.nconstituents >= 2]

    used_indices = set()

    jet_idxs = np.zeros(len(pt), dtype=int)
    for jet_idx, jet in enumerate(jets):
        particle_idx = jet.constituents_idx
        jet_idxs[particle_idx] = jet_idx
        used_indices.update(particle_idx)
    particle_idx = np.arange(len(pt))
    particle_idx = particle_idx[~np.isin(particle_idx, list(used_indices))]
    jet_idxs[particle_idx] = -1

    return jets, jet_idxs


def process_events(
    batch_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
    dl_flag=False,
    dr=0.5,
    eta_cut=3.0,
):
    jetdef = fj.JetDefinition(fj.antikt_algorithm, dr)

    out_data_evt = {
        f"{name}_{var}": []
        for name, var in product(["pflow", "truth", "fastsim"], varlist + ["idx"])
    }

    out_data_evt["eventNumber"] = []

    for name, var in product(
        ["pflow", "truth", "fastsim"], ["pt", "eta", "phi", "d2", "c2"]
    ):
        out_data_evt[f"{name}_jet_{var}"] = []

    for i in range(len(batch_event_numbers)):
        truth_data = {var: evt_truth_data[var][i] for var in varlist}
        pflow_data = {var: evt_pflow_data[var][i] for var in varlist}

        truth_event_number = evt_truth_data["eventNumber"][i]
        pflow_event_number = evt_pflow_data["eventNumber"][i]
        if truth_event_number != pflow_event_number:
            raise ValueError("Event numbers do not match")

        if not dl_flag:
            fs_ind = fs_data["ind"][i].astype(bool)
            fs_evt_data = {var: fs_data[var][i][fs_ind] for var in varlist}
            fs_data["pt"] = fs_data["pt"]  # / 1000
        else:
            fs_evt_data = {var: fs_data[var][i] for var in varlist}

        fs_evt_mask = (
            (fs_evt_data["pt"] > 1)
            & (fs_evt_data["pt"] < 1e4)
            & (np.abs(fs_evt_data["eta"]) <= eta_cut)
        )
        for key, val in fs_evt_data.items():
            fs_evt_data[key] = val[fs_evt_mask]
        pf_evt_mask = np.abs(pflow_data["eta"]) <= eta_cut
        for key, val in pflow_data.items():
            pflow_data[key] = val[pf_evt_mask]

        pf_jets, pf_jet_indices = cluster_jets(
            pflow_data["pt"], pflow_data["eta"], pflow_data["phi"], jetdef
        )
        tr_jets, tr_jet_indices = cluster_jets(
            truth_data["pt"], truth_data["eta"], truth_data["phi"], jetdef
        )
        fs_jets, fs_jet_indices = cluster_jets(
            fs_evt_data["pt"], fs_evt_data["eta"], fs_evt_data["phi"], jetdef
        )

        for key in varlist:
            out_data_evt[f"truth_{key}"].append(truth_data[key])
            out_data_evt[f"pflow_{key}"].append(pflow_data[key])
            out_data_evt[f"fastsim_{key}"].append(fs_evt_data[key])
        out_data_evt["truth_idx"].append(tr_jet_indices)
        out_data_evt["pflow_idx"].append(pf_jet_indices)
        out_data_evt["fastsim_idx"].append(fs_jet_indices)

        for name, jet_collection in zip(
            ["pflow", "truth", "fastsim"], [pf_jets, tr_jets, fs_jets]
        ):
            out_data_evt[f"{name}_jet_pt"].append(
                np.array([jet.pt() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_eta"].append(
                np.array([jet.eta() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_phi"].append(
                np.array([jet.phi() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_d2"].append(
                np.array([jet.substructure["d2"] for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_c2"].append(
                np.array([jet.substructure["c2"] for jet in jet_collection])
            )

        out_data_evt["eventNumber"].append(truth_event_number)
    return out_data_evt


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def process_events_wrapper(args):
    with stdout_redirected():
        return process_events(*args)


def extract_batch_data(
    batch_event_numbers,
    evt_event_numbers,
    fs_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
):
    fs_idx = np.isin(fs_event_numbers, batch_event_numbers)
    evt_idx = np.isin(evt_event_numbers, batch_event_numbers)

    evt_truth_data = {key: val[evt_idx] for key, val in evt_truth_data.items()}
    evt_pflow_data = {key: val[evt_idx] for key, val in evt_pflow_data.items()}
    fs_data = {key: val[fs_idx] for key, val in fs_data.items()}
    return evt_truth_data, evt_pflow_data, fs_data


def main():
    args = parser.parse_args()

    print(f"Loading event file: {args.evt.split('/')[-1]}")
    _, fs_data, fs_event_number = load_file(args.evt, "evt", fs=not args.dl, df=args.df)

    print(f"Loading data file: {'/'.join(args.data.split('/')[-2:])}")
    evt_truth_data, evt_pflow_data, evt_event_number = load_file(
        args.data, "evt", fs=False
    )

    print("All variables loaded")

    out_data_evt = {
        f"{name}_{var}": []
        for name, var in product(["pflow", "truth", "fastsim"], varlist + ["idx"])
    }

    out_data_evt["eventNumber"] = []

    for name, var in product(
        ["pflow", "truth", "fastsim"], ["pt", "eta", "phi", "d2", "c2"]
    ):
        out_data_evt[f"{name}_jet_{var}"] = []

    goodEventNumbers = np.intersect1d(evt_event_number, fs_event_number)

    if args.n_events > len(goodEventNumbers) or args.n_events == -1:
        args.n_events = len(goodEventNumbers)
    print(
        f"Having {len(goodEventNumbers)} good events, processing {args.n_events} of them"
    )

    batch_size = 1000
    n_batches = args.n_events // batch_size
    n_batches += 1 if args.n_events % batch_size != 0 else 0

    input_batches = np.array_split(goodEventNumbers[: args.n_events], n_batches)
    input_batched_data = [
        (
            batch,
            *extract_batch_data(
                batch,
                evt_event_number,
                fs_event_number,
                evt_truth_data,
                evt_pflow_data,
                fs_data,
            ),
            args.dl,
            args.dr,
            args.eta,
        )
        for batch in input_batches
    ]
    # print(input_batched_data[0][3]['ind'].shape, input_batched_data[0][3]['ind'][0].shape)
    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(process_events_wrapper, input_batched_data),
                total=n_batches,
            )
        )

    for result in results:
        for key, val in result.items():
            out_data_evt[key].extend(val)

    output_path = f"{args.output.replace('.root', '')}_{args.n_events // 1000}k_test_eta{int(args.eta * 10)}Eval.root"
    with uproot.recreate(output_path) as f:
        f["evt_tree"] = {
            "truth": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("truth") and "jet" not in key
                }
            ),
            "pflow": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow") and "jet" not in key
                }
            ),
            "fastsim": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("fastsim") and "jet" not in key
                }
            ),
            "eventNumber": out_data_evt["eventNumber"],
            "truth_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("truth_jet") and "idx" not in key
                }
            ),
            "pflow_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow_jet") and "idx" not in key
                }
            ),
            "fastsim_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("fastsim_jet") and "idx" not in key
                }
            ),
        }
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()