import argparse
import os
import sys
import glob

import numpy as np
import awkward as ak

from tqdm import tqdm
from itertools import product

import uproot

sys.path.append("/storage/agrp/dreyet/f_delphes/cms-flow-evt/")
from utils.pdgid import pdgid_class_dict

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


coord_var_list = ["pt", "eta", "phi", "mass"]
vertex_var_list = []
cluster_mom_list = ["widthEta", "widthPhi", "nCells", "EM_PROBABILITY", "CENTER_MAG", "FIRST_ENG_DENS", "CENTER_LAMBDA", "ISOLATION"]

var_dict = {
    "events": ["eventNumber", "mcChannelNumber", "vx", "vy", "vz"],
    "gens": coord_var_list + vertex_var_list + ["pdgId", "status"],
    "pfcs": coord_var_list + vertex_var_list + ["charge"], #"track_idx", "cluster_idx"],
    "tracks": coord_var_list + ["ID", "D0", "Z0"],
    "clusters": ["pt", "eta", "phi", "ID"] + cluster_mom_list,
    "empflow_jets": ["pt", "eta", "phi", "E"],
    "truth_jets": ["pt", "eta", "phi", "E"],
}

# var_dict["mod_pfcs"] = [v for v in var_dict["pfcs"]]

prefix_dict = {
    "gens": "truthPart",
    "pfcs": "Pflow",
    "tracks": "track",
    "clusters": "cluster_",
    "mod_pfcs": "modPflow",
    "empflow_jets": "AntiKt4TruthJets",
    "truth_jets": "AntiKt4TruthJets",
}

replace_dict = {
    "pt": "Pt",
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pdgId": "PdgId",
    "charge": "Charge",
    "vx": "truthVertexX",
    "vy": "truthVertexY",
    "vz": "truthVertexZ",
    "status": "Status",
    "track_idx": "TrackID",
    "cluster_idx": "ClusterID",
}


parser = argparse.ArgumentParser(description="Preprocess full event files")
parser.add_argument(
    "-i", "--input", type=str, required=True, help="Input folder or file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=500
)
parser.add_argument("-pt", "--pt_cut", type=float, help="Truth pt cut", default=0.5)
parser.add_argument("-eta", "--eta_cut", type=float, help="Eta cut", default=2.7)
parser.add_argument("-maxN", "--max_events_total", type=int, help="Max events to process", default=None)
parser.add_argument("-ch", "--chunk_size", type=int, help="Chunk size for writing", default=10000)


def load_files(filelist, n_events=100, entry_start=0, tr_pt_cut=0, eta_cut=2.7, max_events_total=None):
    data = {name: {var: None for var in varlist} for name, varlist in var_dict.items()}
    pbar_files = tqdm(filelist, leave=True, position=0)
    n_events_total = 0
    if max_events_total is None:
        max_events_total = n_events * len(filelist)
    for file_idx, file in enumerate(pbar_files):
        pbar_files.set_description(f"Loading {file.split('/')[-1]}")
        with uproot.open(file) as f:
            tree = f["EventTree"]
            n_events_total += min(n_events, tree.num_entries)
            for name, varlist in var_dict.items():
                for var in varlist:
                    branch = var
                    for new, old in replace_dict.items():
                        branch = branch.replace(new, old)
                    branch = prefix_dict.get(name, "") + branch
                    val = tree[branch].array(
                        library="np",
                        entry_stop=n_events + entry_start,
                        entry_start=entry_start,
                    )
                    if data[name][var] is None:
                        data[name][var] = val
                    else:
                        data[name][var] = np.concatenate((data[name][var], val), axis=0)
        
        if (max_events_total is not None and n_events_total >= max_events_total and file_idx + 1 < len(filelist)):
            print(f"Loaded {n_events_total} events, stopping early, at {file_idx + 1}/{len(filelist)}")
            break

    # derived variables
    charge_to_pdgid = {
        -1: -211,
         0: 22,
         1: 211,
    }
    data["pfcs"]["pdgId"] = np.array(
        [np.vectorize(charge_to_pdgid.get)(charge) for charge in data["pfcs"]["charge"]],
        dtype=object
    )
    if "mod_pfcs" in var_dict:
        data["mod_pfcs"]["pdgId"] = np.array(
            [np.vectorize(charge_to_pdgid.get)(charge) for charge in data["mod_pfcs"]["charge"]],
            dtype=object
        )

    for i in range(len(data["pfcs"]["pt"])):
        pf_mask = (data["pfcs"]["pt"][i] > 1) & (
            np.abs(data["pfcs"]["eta"][i]) < eta_cut
        )
        neutrino_mask = (
            (np.abs(data["gens"]["pdgId"][i]) != 12)
            & (np.abs(data["gens"]["pdgId"][i]) != 14)
            & (np.abs(data["gens"]["pdgId"][i]) != 16)
        )
        status_mask = (
            (data["gens"]["status"][i] == 1)
        )
        tr_mask = (
            (data["gens"]["pt"][i] > tr_pt_cut)
            & (np.abs(data["gens"]["eta"][i]) < eta_cut)
            & neutrino_mask
            & status_mask
        )
        for var in coord_var_list + ["pdgId"]:
            data["pfcs"][var][i] = data["pfcs"][var][i][pf_mask]
            data["gens"][var][i] = data["gens"][var][i][tr_mask]

    return data


def main():
    args = parser.parse_args()
    # check if input is a file or a folder using os.path.isfile
    if os.path.isfile(args.input):
        if args.input.endswith(".root"):
            input_files = [args.input]
        elif args.input.endswith(".txt"):
            with open(args.input, "r") as f:
                input_files = [line.strip() for line in f.readlines()]
        else:
            raise ValueError("Input file must be a .root or .txt file")
    elif os.path.isdir(args.input):
        input_files = glob.glob(args.input + "/*.root")
    data = load_files(
        input_files,
        args.n_events,
        entry_start=args.entry_start,
        tr_pt_cut=args.pt_cut,
        eta_cut=args.eta_cut,
        max_events_total=args.max_events_total,
    )
    print("All variables loaded")

    out_data_evt = {
        f"{name}_{var}": []
        for name, var in product(
            ["pflow", "truth"], coord_var_list + vertex_var_list + ["pdgId", "class"]
        )
    }

    out_data_evt.update({
        f"{name}_jet_{var}": []
        for name, var in product(
            ["truth", "empflow"], ["pt", "eta", "phi", "E"],
        )
    })

    n_events = len(data["events"]["eventNumber"])
    for i in tqdm(range(n_events), desc="Processing events"):

        if len(data["pfcs"]["pt"][i]) == 0 or len(data["gens"]["pt"][i]) == 0:
            continue

        for event_var in var_dict["events"]:
            if event_var in out_data_evt:
                out_data_evt[event_var].append(data["events"][event_var][i])
            else:
                out_data_evt[event_var] = [data["events"][event_var][i]]

        for var in coord_var_list + vertex_var_list + ["pdgId"]:
            out_data_evt["pflow_" + var].append(data["pfcs"][var][i])
            out_data_evt["truth_" + var].append(data["gens"][var][i])

        out_data_evt["pflow_class"].append(
            np.vectorize(pdgid_class_dict.get)(
                out_data_evt["pflow_pdgId"][-1]
            ).astype(int)
        )
        out_data_evt["truth_class"].append(
            np.vectorize(pdgid_class_dict.get)(
                out_data_evt["truth_pdgId"][-1]
            ).astype(int)
        )

        for jet_var in ["pt", "eta", "phi", "E"]:
            out_data_evt["empflow_jet_" + jet_var].append(data["empflow_jets"][jet_var][i])
            out_data_evt["truth_jet_" + jet_var].append(data["truth_jets"][jet_var][i])

    pbar_output = tqdm(range(0, n_events, args.chunk_size))
    for start_idx in pbar_output:
        end_idx = min(start_idx + args.chunk_size, n_events)
        chunk_suffix = f"_{start_idx // args.chunk_size}"
        fname = f"{args.output.replace('.root', '')}_{args.chunk_size // 1000}k{chunk_suffix}.root"
        pbar_output.set_description(f"Writing chunk {chunk_suffix}")
        with uproot.recreate(fname) as f:
            f["evt_tree"] = {
                "eventNumber": ak.Array(out_data_evt["eventNumber"][start_idx:end_idx]),
                "event": ak.zip(
                    {
                        key.split("_")[-1]: ak.Array(val[start_idx:end_idx])
                        for key, val in out_data_evt.items()
                        if key in var_dict["events"] and key != "eventNumber"
                    }
                ),
                "truth": ak.zip(
                    {
                        key.split("_")[-1]: ak.Array(val[start_idx:end_idx])
                        for key, val in out_data_evt.items()
                        if key.startswith("truth") and 'jet' not in key
                    }
                ),
                "pflow": ak.zip(
                    {
                        key.split("_")[-1]: ak.Array(val[start_idx:end_idx])
                        for key, val in out_data_evt.items()
                        if "pflow" in key and 'jet' not in key
                    }
                ),
                "truth_jets": ak.zip(
                    {
                        key.split("_")[-1]: ak.Array(val[start_idx:end_idx])
                        for key, val in out_data_evt.items()
                        if key.startswith("truth_jet")
                    }
                ),
                "empflow_jets": ak.zip(
                    {
                        key.split("_")[-1]: ak.Array(val[start_idx:end_idx])
                        for key, val in out_data_evt.items()
                        if key.startswith("empflow_jet")
                    }
                ),
            }
        print(f"Wrote events {start_idx} to {end_idx} to {fname}")

if __name__ == "__main__":
    main()
