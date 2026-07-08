#!/bin/bash

# Temporary working directory on local disk
TMPDIR="/tmp/ldishman/suep_closure"
mkdir -p "$TMPDIR"

# Output directory for copying final results
OUTDIR="/afs/cern.ch/user/l/ldishman/suep/closure_output/"

# Array of all samples as "Category Mass Temp"
samples=(
#"Leptonic 1.0 1.41" "Leptonic 3.0 4.24" "Leptonic 8.0 22.63"
#"Generic 2.0 0.50"
#"Leptonic 3.0 4.24" "Leptonic 8.0 22.63"
#"Leptonic 5.0 14.14" "Leptonic 6.0 16.97" "Leptonic 7.0 4.95" "Leptonic 7.0 19.80" "Leptonic 8.0 5.66"
#"Generic 2.0 0.71"
#"Leptonic 2.0 0.50" #"Generic 2.0 0.50" "Hadronic 2.0 0.50"
#"Leptonic 3.0 4.24" #"Generic 3.0 4.24" "Hadronic 3.0 4.24"
#"Leptonic 5.0 10.00" # "Generic 5.0 10.00" "Hadronic 5.0 10.00"
"Leptonic 8.0 22.63" # "Generic 8.0 22.63" "Hadronic 8.0 22.63" 
#"Leptonic 2.0 0.71" # "Hadronic 2.0 0.71"

#"Generic 5.0 7.07"   "Hadronic 1.40 3.96"  "Hadronic 5.0 3.54"   "Leptonic 1.0 1.41"   "Leptonic 5.0 20.00"
#"Generic 6.0 12.00"  "Hadronic 1.40 5.60"  "Hadronic 5.0 5.00"   "Leptonic 1.0 2.00"   "Leptonic 5.0 2.50"
#"Generic 2.0 1.00"   "Generic 6.0 1.50"    "Hadronic 5.0 7.07"   "Leptonic 1.0 2.83"   "Leptonic 5.0 3.54"
#"Generic 2.0 1.41"   "Generic 6.0 16.97"   "Hadronic 6.0 12.00"  "Leptonic 1.0 4.00"   "Leptonic 5.0 5.00"
#"Generic 2.0 2.00"   "Generic 6.0 2.12"    "Hadronic 2.0 1.00"    "Hadronic 6.0 1.50"   "Leptonic 5.0 7.07"
#"Generic 2.0 2.83"   "Generic 6.0 24.00"   "Hadronic 2.0 1.41"    "Hadronic 6.0 16.97"  "Leptonic 6.0 12.00"
#"Generic 2.0 4.00"   "Generic 6.0 3.00"    "Hadronic 2.0 2.00"    "Hadronic 6.0 2.12"   "Leptonic 2.0 1.00"   "Leptonic 6.0 1.50"
#"Generic 2.0 5.66"   "Generic 6.0 4.24"    "Hadronic 2.0 2.83"    "Hadronic 6.0 24.00"  "Leptonic 2.0 1.41"   "Leptonic 6.0 16.97"
#"Generic 2.0 8.00"   "Generic 6.0 6.00"    "Hadronic 2.0 4.00"    "Hadronic 6.0 3.00"   "Leptonic 2.0 2.00"   "Leptonic 6.0 2.12"
#"Generic 3.0 0.75"   "Generic 6.0 8.49"    "Hadronic 2.0 5.66"    "Hadronic 6.0 4.24"   "Leptonic 2.0 2.83"   "Leptonic 6.0 24.00"
#"Generic 3.0 1.06"   "Generic 7.0 14.00"   "Hadronic 2.0 8.00"    "Hadronic 6.0 6.00"   "Leptonic 2.0 4.00"   "Leptonic 6.0 3.00"
#"Generic 3.0 12.00"  "Generic 7.0 1.75"    "Hadronic 3.0 0.75"    "Hadronic 6.0 8.49"   "Leptonic 2.0 5.66"   "Leptonic 6.0 4.24"
#"Generic 3.0 1.50"   "Generic 7.0 19.80"   "Hadronic 3.0 1.06"    "Hadronic 7.0 14.00"  "Leptonic 2.0 8.00"   "Leptonic 6.0 6.00"
#"Generic 3.0 2.12"   "Generic 7.0 2.47"    "Hadronic 3.0 12.00"   "Hadronic 7.0 1.75"   "Leptonic 3.0 0.75"   "Leptonic 6.0 8.49"
#"Generic 3.0 3.00"   "Generic 7.0 28.00"   "Hadronic 3.0 1.50"    "Hadronic 7.0 19.80"  "Leptonic 3.0 1.06"   "Leptonic 7.0 14.00"
#"Generic 7.0 3.50"    "Hadronic 3.0 2.12"    "Hadronic 7.0 2.47"   "Leptonic 3.0 12.00"  "Leptonic 7.0 1.75"
#"Generic 3.0 6.00"   "Generic 7.0 4.95"    "Hadronic 3.0 3.00"    "Hadronic 7.0 28.00"  "Leptonic 3.0 1.50"   "Leptonic 7.0 19.80"
#"Generic 3.0 8.49"   "Generic 7.0 7.00"    "Hadronic 7.0 3.50"   "Leptonic 3.0 2.12"   "Leptonic 7.0 2.47"
#"Generic 4.0 1.00"   "Generic 7.0 9.90"    "Hadronic 3.0 6.00"    "Hadronic 7.0 4.95"   "Leptonic 3.0 3.00"   "Leptonic 7.0 28.00"
#"Generic 4.0 11.31"  "Generic 8.0 11.31"   "Hadronic 3.0 8.49"    "Hadronic 7.0 7.00"   "Leptonic 7.0 3.50"
#"Generic 4.0 1.41"   "Generic 8.0 16.00"   "Hadronic 4.0 1.00"    "Hadronic 7.0 9.90"   "Leptonic 3.0 6.00"   "Leptonic 7.0 4.95"
#"Generic 4.0 16.00"  "Generic 8.0 2.00"    "Hadronic 4.0 11.31"   "Hadronic 8.0 11.31"  "Leptonic 3.0 8.49"   "Leptonic 7.0 7.00"
#"Generic 4.0 2.00"   "Hadronic 4.0 1.41"    "Hadronic 8.0 16.00"  "Leptonic 4.0 1.00"   "Leptonic 7.0 9.90"
#"Generic 4.0 2.83"   "Generic 8.0 2.83"    "Hadronic 4.0 16.00"   "Hadronic 8.0 2.00"   "Leptonic 4.0 11.31"  "Leptonic 8.0 11.31"
#"Generic 4.0 4.00"   "Generic 8.0 32.00"   "Hadronic 4.0 2.00"    "Leptonic 4.0 1.41"   "Leptonic 8.0 16.00"
#"Generic 4.0 5.66"   "Generic 8.0 4.00"    "Hadronic 4.0 2.83"    "Hadronic 8.0 2.83"   "Leptonic 4.0 16.00"  "Leptonic 8.0 2.00"
#"Generic 4.0 8.00"   "Generic 8.0 5.66"    "Hadronic 4.0 4.00"    "Hadronic 8.0 32.00"  "Leptonic 4.0 2.00"   
#"Generic 8.0 8.00"    "Hadronic 4.0 5.66"    "Hadronic 8.0 4.00"   "Leptonic 4.0 2.83"   "Leptonic 8.0 2.83"
#"Generic 5.0 1.25"   "Hadronic 1.40 0.35"  "Hadronic 4.0 8.00"    "Hadronic 8.0 5.66"   "Leptonic 4.0 4.00"   "Leptonic 8.0 32.00"
#"Generic 5.0 14.14"  "Hadronic 1.40 0.49"  "Hadronic 8.0 8.00"   "Leptonic 4.0 5.66"   "Leptonic 8.0 4.00"
#"Generic 5.0 1.77"   "Hadronic 1.40 0.70"  "Hadronic 5.0 1.25"    "Leptonic 1.0 0.25"   "Leptonic 4.0 8.00"   "Leptonic 8.0 5.66"
#"Generic 5.0 20.00"  "Hadronic 1.40 0.99"  "Hadronic 5.0 14.14"   "Leptonic 1.0 0.35"   "Leptonic 8.0 8.00"
#"Generic 5.0 2.50"   "Hadronic 1.40 1.40"  "Hadronic 5.0 1.77"    "Leptonic 1.0 0.50"   "Leptonic 5.0 1.25"
#"Generic 5.0 3.54"   "Hadronic 1.40 1.98"  "Hadronic 5.0 20.00"   "Leptonic 1.0 0.71"   "Leptonic 5.0 14.14"
#"Generic 5.0 5.00"   "Hadronic 1.40 2.80"  "Hadronic 5.0 2.50"    "Leptonic 1.0 1.00"   "Leptonic 5.0 1.77"
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
