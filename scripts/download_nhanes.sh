#!/bin/bash
# Downloads the NHANES files this project uses, straight from the CDC website,
# plus each file's codebook page, and writes data/raw/manifest.csv (SHA-256).
#
# Run on a machine that can reach wwwn.cdc.gov:
#     bash scripts/download_nhanes.sh
# Re-running is safe: valid files already downloaded are skipped.
# Needs only bash and curl (built into macOS and Linux).

set -u
cd "$(dirname "$0")/.." || exit 1

BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
RAW="data/raw"
DOCS="data/codebooks"
UA="Mozilla/5.0 (ES335 course project; NHANES download)"
COMPONENTS="DEMO GHB DIQ BMX BPQ MCQ RHQ PAQ SMQ WHQ GLU HDL"

# cycle label | year folder on the CDC site | file-name pattern (%s = component)
CYCLES="2007-2008|2007|%s_E
2009-2010|2009|%s_F
2011-2012|2011|%s_G
2013-2014|2013|%s_H
2015-2016|2015|%s_I
2017-2020|2017|P_%s"

if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

is_xpt() { [ -s "$1" ] && head -c 40 "$1" | grep -q "HEADER RECORD"; }

fetch() {  # fetch URL DEST
  curl -fsSL --retry 4 --retry-delay 5 --connect-timeout 30 --max-time 900 \
       -A "$UA" -o "$2.part" "$1" && mv "$2.part" "$2"
}

ok=0; failed=""; start=$(date +%s)
echo "Downloading NHANES files into $RAW ..."
while IFS='|' read -r label year pattern; do
  [ -z "$label" ] && continue
  mkdir -p "$RAW/$label" "$DOCS/$label"
  for comp in $COMPONENTS; do
    name=$(printf "$pattern" "$comp")
    dest="$RAW/$label/$name.xpt"
    if is_xpt "$dest"; then
      echo "  have   $label  $name"
    else
      got=0
      for ext in xpt XPT; do
        if fetch "$BASE/$year/DataFiles/$name.$ext" "$dest" && is_xpt "$dest"; then got=1; break; fi
        rm -f "$dest" "$dest.part"
      done
      if [ $got -eq 1 ]; then
        echo "  got    $label  $name  ($(wc -c < "$dest" | tr -d ' ') bytes)"
      else
        echo "  FAILED $label  $name"; failed="$failed $label/$name"; continue
      fi
    fi
    ok=$((ok+1))
    page="$DOCS/$label/$name.htm"
    [ -s "$page" ] || fetch "$BASE/$year/DataFiles/$name.htm" "$page" || echo "  (no codebook page for $name)"
  done
done <<EOF
$CYCLES
EOF

now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo "cycle,file,bytes,sha256,source_url,checked_utc"
  for f in "$RAW"/*/*.xpt; do
    [ -e "$f" ] || continue
    label=$(basename "$(dirname "$f")"); name=$(basename "$f" .xpt); year=${label%%-*}
    echo "$label,$name.xpt,$(wc -c < "$f" | tr -d ' '),$($SHA "$f" | cut -d' ' -f1),$BASE/$year/DataFiles/$name.xpt,$now"
  done
} > "$RAW/manifest.csv"

echo
echo "Done in $(( $(date +%s) - start )) s: $ok of 72 files present."
if [ -n "$failed" ]; then echo "Failed:$failed  (run the script again to retry)"; exit 1; fi
echo "All files downloaded. Manifest: $RAW/manifest.csv"
