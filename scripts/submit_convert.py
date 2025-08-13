import os
import glob

indir = "/storage/agrp/dreyet/MLTreeAthenaAnalysis/output"
outdir = "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS"

indirs = {
    "JZ1": f"{indir}/user.edreyer.801166.Py8EG_A14NNPDF23LO_jj_JZ1.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ2": f"{indir}/user.edreyer.801167.Py8EG_A14NNPDF23LO_jj_JZ2.recon.ESD.e8514_e8528_s4186_s4114_r15440_05012025_mltree.root",
    "JZ3": f"{indir}/user.edreyer.801168.Py8EG_A14NNPDF23LO_jj_JZ3.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ4": f"{indir}/user.edreyer.801169.Py8EG_A14NNPDF23LO_jj_JZ4.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ5": f"{indir}/user.edreyer.801170.Py8EG_A14NNPDF23LO_jj_JZ5.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ6": f"{indir}/user.edreyer.801171.Py8EG_A14NNPDF23LO_jj_JZ6.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ7": f"{indir}/user.edreyer.801172.Py8EG_A14NNPDF23LO_jj_JZ7.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
    "JZ8": f"{indir}/user.edreyer.801173.Py8EG_A14NNPDF23LO_jj_JZ8.recon.ESD.e8514_e8528_s4185_s4114_r14977_05012025_mltree.root",
}

def write_text_files(pattern, output, max_lines=100):
    """
    Glob the files matching the pattern and write them in chunks of max_lines or less to text files
    """
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    file_lists = []
    if len(files) <= max_lines:
        file_lists.append(files)
    else:
        for i in range(0, len(files), max_lines):
            file_lists.append(files[i:i + max_lines])

    output_files = []
    for i, file_list in enumerate(file_lists):
        output_file = output.replace(".txt", f"_{i}.txt")
        with open(output_file, "w") as f:
            for file in file_list:
                f.write(file + "\n")
        print(f"# Wrote {len(file_list)} files to {output_file}")
        output_files.append(output_file)

    return output_files

def get_job_command(file):

    log = file.replace(".txt", ".out")
    err = file.replace(".txt", ".err")
    out = file.replace(".txt", ".root")
    payload = "/storage/agrp/dreyet/f_delphes/cms-flow-evt/run_convert.sh"

    cmd = f"qsub -o {log} -e {err} -q N -N converter -l walltime=1:59:00,mem=6gb,ncpus=1,io=6 -v INFILE={file},OUTFILE={out} {payload}"

    return cmd

def main():
    # Loop over the keys in the dictionary
    jobs = []
    for key, value in indirs.items():
        # Construct the full path to the file
        pattern = os.path.join(indir, value)
        pattern += "/*.root"
        # Construct the output file name
        output = os.path.join(outdir, f"{key}.txt")
        # Call the function to write the text files
        text_files = write_text_files(pattern, output)
        # Loop over the text files and submit jobs
        for tf in text_files:
            job = get_job_command(tf)
            jobs.append(job)

    # Write the jobs to a file
    with open("submit_convert.sh", "w") as f:
        for job in jobs:
            f.write(job + "\n")

    print(f"# Wrote {len(jobs)} jobs to submit_convert.sh")

if __name__ == "__main__":
    main()
