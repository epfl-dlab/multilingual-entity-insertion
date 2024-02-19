
set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
git clone https://github.com/WenzhengZhang/EntQA
DST_DIR="$ROOD_DIR/EntQA"
cd "$DST_DIR"

mkdir -p "models"
cd "models"
wget https://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json
gdown 1bHS5rxGbHJ5omQ-t8rjQogw7QJq-qYFO

cd "$ROOD_DIR"