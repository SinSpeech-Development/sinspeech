#!/usr/bin/env bash

start=$(date +%s.%N)

stage=0
LOG_LOCATION=`pwd`/logs

model=$1

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"decode_test_""$model".log 2>&1

nj=$(nproc)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

# train a monophone system
if [[ $stage -le 1 && "$model" == "mono" ]]; then

    echo "===== BEGIN : mono Test set decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/mono/graph data/test \
                          exp/mono/decode_test
    echo
    echo "===== END: mono Test set decode ====="

fi

# train a first delta + delta-delta triphone system on all utterances
if [[ $stage -le 2 && "$model" == "tri1" ]]; then

    echo "===== BEGIN : tri1 Test set decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri1/graph data/test \
                          exp/tri1/decode_test
    echo
    echo "===== END: tri1 Test set decode ====="
    
fi

# train an LDA+MLLT system.
if [[ $stage -le 3 && "$model" == "tri2" ]]; then

    echo "===== BEGIN : tri2b Test set decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri2b/graph data/test \
                          exp/tri2b/decode_test
    echo
    echo "===== END: tri2b Test set decode ====="

fi

# Train tri3b, which is LDA+MLLT+SAT
if [[ $stage -le 4 && "$model" == "tri3" ]]; then

    echo "===== BEGIN : tri3b fmllr Test set decode ====="
    echo
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph data/test \
                          exp/tri3b/decode_test
    echo
    echo "===== END: tri3b fmllr Test set decode ====="
    
fi

end=$(date +%s.%N)    
runtime=$(python -c "print(${end} - ${start})")

echo "Runtime was $runtime"

exit 0