#!/bin/bash

# NOTE: with the rewritten stat_mix.py this whole driver is obsolete.
# stat_mix.py now takes NO arguments, has no held-out target, and writes
# ONE shapes.npz (the mix f_s is identical for every target). Nothing to
# loop over, nothing to copy from /tmp. Just run:  python3 stat_mix.py

# Temporary working directory on local disk (must match tmpdir in stat_mix.py)
TMPDIR="/tmp/ldishman/suep_variance"
mkdir -p "$TMPDIR"

# Output directory for collecting the per-target shape dumps
OUTDIR="mix_output"
mkdir -p "$OUTDIR"

# Array of ALL targets you want guaranteed <= 0.1%, as "Category Mass Temp"
samples=(
"Hadronic 1.40 3.96"
"Hadronic 1.40 5.60"
"Hadronic 1.40 0.35"
"Hadronic 1.40 0.49"
"Hadronic 1.40 0.70"
"Hadronic 1.40 0.99"
"Hadronic 1.40 1.40"
"Hadronic 1.40 1.98"
"Hadronic 1.40 2.80"
#"Generic 5.0 5.00" "Hadronic 5.0 2.50"    "Leptonic 1.0 1.00"   "Leptonic 5.0 1.77"
)

# Loop over all targets and run the variance script
for s in "${samples[@]}"; do
    read category m T <<< "$s"
    python3 stat_mix.py "$category" "$m" "$T"

    # Copy this target's output (incl. shapes.npz) into mix_output
    prefix="variance_${category}_m$(printf "%.1f" "$m")_T$(printf "%.2f" "$T")_"
    #prefix="variance_${category}_m$(printf "%.2f" "$m")_T$(printf "%.2f" "$T")_"
    cp -r "$TMPDIR/${prefix}output" "$OUTDIR/"
done

# Cleanup /tmp
rm -rf "$TMPDIR"
