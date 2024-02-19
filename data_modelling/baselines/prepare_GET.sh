set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
git clone https://github.com/facebookresearch/GROOV.git
DST_DIR="$ROOD_DIR/GROOV"
cd "$DST_DIR"
mkdir -p "model_checkpoint"
wget https://dl.fbaipublicfiles.com/groov/get_model.tar.gz
tar -xvf get_model.tar.gz -C model_checkpoint
rm get_model.tar.gz
cd "$ROOD_DIR"