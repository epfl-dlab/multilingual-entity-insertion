# Description: This script is used to deploy a Run:AI job to a Kubernetes cluster.
# Usage: bash runai_deploy.sh

# Set the following environment variables:
export IMAGE_CREATOR="wendler"
export IMAGE_NAME="conda"
export IMAGE_TAG="prod"
export IMAGE="ic-registry.epfl.ch/dlab/${IMAGE_CREATOR}/${IMAGE_NAME}:${IMAGE_TAG}"
export NODE_TYPE="G10"
export NUM_GPUS=1
export NUM_CPUS=4
export MEMORY="48G"
export SLEEP_TIME="24h"
# for LANG in "af" "cs" "cy" "en" "fr" "ga" "gu" "hi" "is" "it" "ja" "kk" "kn" "ms" "ps" "pt" "simple" "sk" "sw" "ur" "uz"
# for LANG in "en" "de" "fr" "sv" "nl" "ru" "es" "it" "pl" "zh" "ja" "vi" "uk" "war" "ar" "pt" "fa" "ca" "id" "sr" "ko" "no" "ce" "fi" "tr" "cs" "hu" "tt" "sh" "ro" "eu" "ms" "eo" "he" "da" "hy" "bg" "cy" "sk" "uz" "azb" "be" "et" "simple" "kk" "el" "min" "hr" "lt" "az" "gl" "ur" "sl" "ka" "hi" "ta" "la" "mk" "ast" "lv" "af" "tg" "mg" "sq" "mr" "bs" "oc" "te" "br" "sw" "ku" "lmo" "jv" "pms" "ba" "lb" "su" "ga" "is" "cv" "fy" "pa" "tl" "io" "an" "vo" "ha" "sco" "ne" "kn" "gu" "bar" "scn" "mn" "si" "ps" "gd" "am" "sd" "yi" "as" "sa" "km" "ary" "nds-nl" "so" "ug" "lo" "xh" "om"
# for LANG in "af" "cs" "cy" "en" "ga" "gu" "hi" "is" "it" "ja" "kk" "kn" "ms" "ps" "pt" "simple" "sw" "ur" "uz"
# for LANG in "fr" "sk"
for LANG in "simple"
do
    export LANGUAGE="${LANG}"
    EXP_NAME="runai-${IMAGE_NAME}-${IMAGE_TAG}-${SLEEP_TIME}-${NODE_TYPE}-${NUM_GPUS}-${NUM_CPUS}-${MEMORY}-${LANG}-benchmark"
    # lowercase EXP_NAME, because k8s doesn't allow uppercase letters in the name
    # shellcheck disable=SC2155
    export EXP_NAME=$(echo "$EXP_NAME" | tr '[:upper:]' '[:lower:]')

    # Save the new deployment file in the history directory
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR="history/${IMAGE_NAME}/${EXP_NAME}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    envsubst < runai_deploy_conf.yaml > "${OUTPUT_DIR}/config.yaml"

    # Save the deploy script in the history directory
    cat runai_deploy_benchmark.sh > "${OUTPUT_DIR}/deploy.sh"

    # Apply the deployment file to the cluster
    kubectl create -f "$OUTPUT_DIR/config.yaml"
done

# Reset the environment variables
unset IMAGE_NAME
unset EXP_NAME
unset IMAGE
unset NODE_TYPE
unset NUM_GPUS
unset NUM_CPUS
unset MEMORY
unset SLEEP_TIME
unset TIMESTAMP
unset OUTPUT_DIR
unset LANG