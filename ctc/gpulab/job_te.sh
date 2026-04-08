#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --job-name=morse-seq-ctc
#SBATCH --partition=gpu-ffa
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100:1
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

DATA_DIR="/home/bulirma/ctc"
WORK_DIR="$TMPDIR"

cp "${DATA_DIR}/traineval.py" "$WORK_DIR/"
cp "${DATA_DIR}/models.py" "$WORK_DIR/"
cp "${DATA_DIR}/dataset.py" "$WORK_DIR/"
cp "${DATA_DIR}/morse-sequence.pklz" "$WORK_DIR/"

cd "$WORK_DIR" || exit 1

. "${DATA_DIR}/venv/bin/activate"

python traineval.py --dataset morse-sequence.pklz

EXIT=$?

deactivate

cp -r "$WORK_DIR/dev-models" "${DATA_DIR}/"
rm -rf "$WORK_DIR"

exit $EXIT
