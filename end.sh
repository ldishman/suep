#!/bin/bash

# Temporary working directory on local disk
TMPDIR="/tmp/ldishman/suep_closure"
mkdir -p "$TMPDIR"

# Output directory for copying final results
OUTDIR="/afs/cern.ch/user/l/ldishman/suep/closure_output/"

# Array of all samples as "Category Mass Temp"
samples=(
#"Leptonic 1.0 1.41" "Leptonic 3.0 4.24" "Leptonic 8.0 22.63"
#"Leptonic 5.0 14.14" "Leptonic 6.0 16.97" "Leptonic 7.0 19.80"
"Leptonic 2.0 0.50"
)

# Loop over all samples and run Python script
for s in "${samples[@]}"; do
    read category m T <<< "$s"
    python3 closure.py "$category" "$m" "$T"

    # Copy this sample's output immediately to AFS
    prefix="closure_${category}_m$(printf "%.1f" "$m")_T$(printf "%.2f" "$T")_"
    cp -r "$TMPDIR/${prefix}output" "$OUTDIR/"
done

# Cleanup /tmp
rm -rf "$TMPDIR"
