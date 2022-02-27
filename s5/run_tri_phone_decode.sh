#!/usr/bin/env bash

stage=0
LOG_LOCATION=`pwd`/logs

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"run_tri_phone_decode"$stage.log 2>&1

nj=$(nproc)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
    
    # decode using the tri2b model
    echo "===== BEGIN : make tri2b graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri2b exp/tri2b/graph
    echo
    echo "===== END: make tri2b graph ====="

    # decode using the tri3b model
    echo "===== BEGIN : make tri3b graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri3b exp/tri3b/graph
    echo
    echo "===== END: make tri3b graph ====="
fi

if [ $stage -le 1 ]; then

    echo "===== BEGIN : tri2b decode test ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri2b/graph data/test \
                          exp/tri2b/decode_test
    echo
    echo "===== END: tri2b decode test ====="

fi

if [ $stage -le 2 ]; then

    echo "===== BEGIN : tri3b fmllr decode test ====="
    echo
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph data/test \
                          exp/tri3b/decode_test
    echo
    echo "===== END: tri3b fmllr decode test ====="
fi
