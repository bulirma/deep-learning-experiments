#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --job-name=ctc_morse_traineval
#SBATCH --partition=gpu-ffa
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:2
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

DATA_DIR="/home/bulirma/ctc"
WORK_DIR="$TMPDIR"

cp "${DATA_DIR}/traineval.py" "$WORK_DIR/"
cp "${DATA_DIR}/models.py" "$WORK_DIR/"
cp "${DATA_DIR}/datasets.py" "$WORK_DIR/"
cp "${DATA_DIR}/morse-dataset.pklz" "$WORK_DIR/"

cd "$WORK_DIR" || exit 1

. "${DATA_DIR}/venv/bin/activate"

python traineval.py

EXIT=$?

deactivate

cp -r "$WORK_DIR/models" "${DATA_DIR}/"
rm -rf "$WORK_DIR"

exit $EXIT
