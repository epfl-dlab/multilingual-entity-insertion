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
export MEMORY="64G"
export SLEEP_TIME="24h"

for MODEL_NAME_ITER in "bert-base-cased" "roberta-base"
# for MODEL_NAME_ITER in "t5-base"
do
    for BASH_FILE_ITER in "runai-simple-finetuning" "runai-dyn-mask" "runai-section-knowledge" "runai-base-plus-second-stage" "runai-only-second-stage" "runai-no-corrupt-plus-second-stage"
    # for BASH_FILE_ITER in "runai-no-corrupt-plus-second-stage"
    # for BASH_FILE_ITER in "runai-large-list"
    do
        # set the architecture based on the model name
        if [ "$MODEL_NAME_ITER" = "bert-base-cased" ]; then
            export MODEL_ARCHITECTURE="BERT"
            export BATCH_SIZE=5
        elif [ "$MODEL_NAME_ITER" = "roberta-base" ]; then
            export MODEL_ARCHITECTURE="RoBERTa"
            export BATCH_SIZE=5
        elif [ "$MODEL_NAME_ITER" = "t5-base" ]; then
            export MODEL_ARCHITECTURE="T5"
            export BATCH_SIZE=4
        fi
        export MODEL_NAME="${MODEL_NAME_ITER}"
        # add the .sh extension to the bash file
        export BASH_FILE="${BASH_FILE_ITER}.sh"
        EXP_NAME="runai-${IMAGE_NAME}-${IMAGE_TAG}-${SLEEP_TIME}-${NODE_TYPE}-${NUM_GPUS}-${NUM_CPUS}-${MEMORY}-${MODEL_NAME_ITER}-${BASH_FILE_ITER}"
        # lowercase EXP_NAME, because k8s doesn't allow uppercase letters in the name
        # shellcheck disable=SC2155
        export EXP_NAME=$(echo "$EXP_NAME" | tr '[:upper:]' '[:lower:]')

        # Save the new deployment file in the history directory
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        OUTPUT_DIR="history/${IMAGE_NAME}/${EXP_NAME}_${TIMESTAMP}"
        mkdir -p "$OUTPUT_DIR"
        envsubst < runai_deploy_conf.yaml > "${OUTPUT_DIR}/config.yaml"

        # Save the deploy script in the history directory
        cat runai_deploy_multiple.sh > "${OUTPUT_DIR}/deploy.sh"

        # Apply the deployment file to the cluster
        kubectl create -f "$OUTPUT_DIR/config.yaml"
    done
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
unset MODEL_NAME
unset BASH_FILE
unset MODEL_ARCHITECTURE
