#!/usr/bin/env bash
# =============================================================================
# convert_checkpoint.sh — Convert VERL FSDP shards → HuggingFace safetensors
#
# Usage:
#   bash scripts/convert_checkpoint.sh \
#       --checkpoint_dir /path/to/global_step_400/actor \
#       --output_dir     /path/to/global_step_400/actor_hf
#
# After conversion, verify with:
#   python -c "
#   from transformers import AutoModelForCausalLM
#   m = AutoModelForCausalLM.from_pretrained('/path/to/actor_hf')
#   print('OK:', sum(p.numel() for p in m.parameters()), 'params')
#   "
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --checkpoint_dir and --output_dir are both required."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

cd "$PROJECT_DIR"

echo "Converting FSDP checkpoint:"
echo "  From: $CHECKPOINT_DIR"
echo "  To:   $OUTPUT_DIR"
echo ""

python convert_checkpoint.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir     "$OUTPUT_DIR"

echo ""
echo "Verifying output ..."
python - <<PYEOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

out_dir = "$OUTPUT_DIR"
try:
    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    m   = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)
    n   = sum(p.numel() for p in m.parameters())
    print(f"✓ Loaded successfully: {n/1e9:.2f}B parameters, vocab={len(tok)}")
except Exception as e:
    print(f"✗ Verification failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
