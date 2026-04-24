#!/bin/bash

# Temporary working directory on local disk
TMPDIR="/tmp/ldishman/suep_job"
mkdir -p "$TMPDIR"

# Output directory for copying final results
OUTDIR="/afs/cern.ch/user/l/ldishman/suep/refM3_output/"

# Array of all samples as "Category Mass Temp"
samples=(
"Leptonic 4.0 1.00"   "Leptonic 4.0 11.31"   "Leptonic 4.0 1.41"   "Leptonic 4.0 16.00"   "Leptonic 4.0 2.83"   "Leptonic 4.0 5.66"   "Leptonic 4.0 8.00"
)

# Loop over all samples and run Python script
for s in "${samples[@]}"; do
    read category m T <<< "$s"
    python3 test.py "$category" "$m" "$T" 

    # Copy this sample’s output immediately to AFS
    prefix="${category}_m$(printf "%.1f" "$m")_T$(printf "%.2f" "$T")_"
    cp -r "$TMPDIR/${prefix}output" "$OUTDIR/"
done

# Cleanup /tmp
rm -rf "$TMPDIR"
