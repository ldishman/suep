#!/bin/bash

# Temporary working directory on local disk
TMPDIR="/tmp/ldishman/suep_job"
mkdir -p "$TMPDIR"

# Output directory for copying final results
OUTDIR="/afs/cern.ch/user/l/ldishman/suep/refM3_output/"

# Array of all samples as "Category Mass Temp"
samples=(
#"Leptonic 1.0 0.50" "Leptonic 2.0 0.50" "Leptonic 2.0 2.00" "Leptonic 3.0 0.75" "Leptonic 5.0 1.25" "Leptonic 5.0 2.50" "Leptonic 6.0 1.50" "Leptonic 6.0 6.00" "Leptonic 6.0 12.00" "Leptonic 7.0 14.00" "Leptonic 8.0 2.00"
"Leptonic 1.0 0.35" "Leptonic 1.0 0.71" "Leptonic 2.0 0.71" "Leptonic 2.0 1.41" "Leptonic 2.0 2.83" "Leptonic 5.0 1.77" "Leptonic 5.0 7.07" "Leptonic 5.0 14.14" "Leptonic 6.0 2.12" "Leptonic 6.0 16.97" "Leptonic 7.0 2.47" "Leptonic 7.0 4.95" "Leptonic 7.0 9.90" "Leptonic 8.0 2.83" "Leptonic 8.0 5.66" "Leptonic 8.0 11.31" "Leptonic 8.0 22.63"
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
