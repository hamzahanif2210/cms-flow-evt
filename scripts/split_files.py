import ROOT
import glob
import random
import os

random.seed(42)

all_files = glob.glob("JZ*_10k_*.root")
groups = {}
for f in all_files:
    key = f.split("_")[0]   # e.g. JZ1
    groups.setdefault(key, []).append(f)

os.makedirs("splits", exist_ok=True)

train_files, val_files, test_files = [], [], []

for key, files in groups.items():

    print(f"Processing {key} with {len(files)} files")
    df = ROOT.RDataFrame("evt_tree", files)  # replace "tree"
    nEntries = df.Count().GetValue()
    
    indices = list(range(nEntries))
    random.shuffle(indices)
    
    n_train = int(0.85 * nEntries)
    n_val   = int(0.10 * nEntries)
    
    df = df.Define("entryIdx", "rdfentry_")
    train_df = df.Filter(f"entryIdx < {n_train}")
    val_df   = df.Filter(f"entryIdx >= {n_train} && entryIdx < {n_train+n_val}")
    test_df  = df.Filter(f"entryIdx >= {n_train+n_val}")
    
    # Write intermediate split files
    f_train = f"splits/{key}_train.root"
    f_val   = f"splits/{key}_val.root"
    f_test  = f"splits/{key}_test.root"
    
    train_df.Snapshot("evt_tree", f_train)
    val_df.Snapshot("evt_tree", f_val)
    test_df.Snapshot("evt_tree", f_test)
    
    train_files.append(f_train)
    val_files.append(f_val)
    test_files.append(f_test)

# Use TChain to merge into final files
def merge_files(files, outname):
    chain = ROOT.TChain("evt_tree")
    for f in files:
        chain.Add(f)
    df = ROOT.RDataFrame(chain)
    df.Snapshot("evt_tree", outname)

merge_files(train_files, "train.root")
merge_files(val_files, "val.root")
merge_files(test_files, "test.root")

print("✅ Finished: train.root, val.root, test.root")
