#!/usr/bin/env bash

start=$(date +%s.%N)

stage=0
LOG_LOCATION=`pwd`/logs

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"run_phones_"$stage.log 2>&1

nj=$(nproc)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then

  # Get the shortest 5000 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 5000 data/train_5000short

fi

# train a monophone system
if [ $stage -le 1 ]; then
    echo "===== BEGIN : Train 5000 Short Mono ====="
    echo
    # TODO(galv): Is this too many jobs for a smaller dataset?
    steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/train_5000short data/lang exp/mono
    echo
    echo "===== END: Train 5000 Short Mono ====="

    echo "===== BEGIN : mono align ====="
    echo
    steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/mono exp/mono_ali_train
    echo
    echo "===== END: mono align ====="
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 2 ]; then
    echo "===== BEGIN : Train delta + delta-delta triphone ====="
    echo
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
        2000 10000 data/train data/lang exp/mono_ali_train exp/tri1
    echo
    echo "===== END: Train delta + delta-delta triphone ====="

    echo "===== BEGIN : tri1 align ====="
    echo
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri1 exp/tri1_ali_train
    echo
    echo "===== END: tri1 align ====="
fi

# train an LDA+MLLT system.
if [ $stage -le 3 ]; then
    echo "===== BEGIN: train LDA+MLLT - tri2b model ====="
    echo
    steps/train_lda_mllt.sh  --cmd "$train_cmd" \
        --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
        data/train data/lang exp/tri1_ali_train exp/tri2b
    echo
    echo "===== END: train LDA+MLLT - tri2b model ====="

    echo "===== BEGIN: tri2b model align ====="
    echo
    # Align utts using the tri2b model
    steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
        data/train data/lang exp/tri2b exp/tri2b_ali_train
    echo
    echo "===== END: tri2b model align ====="
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 4 ]; then
    echo "===== BEGIN : train LDA+MLLT+SAT - tri3b model ====="
    echo
    steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
        data/train data/lang exp/tri2b_ali_train exp/tri3b
    echo
    echo "===== END: train LDA+MLLT+SAT - tri3b model ====="

    echo "===== BEGIN: tri3b model align ====="
    echo
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri3b exp/tri3b_ali_train
    echo
    echo "===== END: tri3b model align ====="
fi


if [ $stage -le 5 ]; then
    # Test the tri3b system with the silprobs and pron-probs.

    # decode using the mono model
    echo "===== BEGIN : make mono graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/mono exp/mono/graph
    echo
    echo "===== END: make mono graph ====="

    # decode using the tri1 model
    echo "===== BEGIN : make tri1 graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri1 exp/tri1/graph
    echo
    echo "===== END: make tri1 graph ====="

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

if [ $stage -le 6 ]; then

    echo "===== BEGIN : mono decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/mono/graph data/test \
                          exp/mono/decode_test
    echo
    echo "===== END: mono decode ====="

    echo "===== BEGIN : tri1 decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri1/graph data/test \
                          exp/tri1/decode_test
    echo
    echo "===== END: tri1 decode ====="

    echo "===== BEGIN : tri2b decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri2b/graph data/test \
                          exp/tri2b/decode_test
    echo
    echo "===== END: tri2b decode ====="

    echo "===== BEGIN : tri3b fmllr decode ====="
    echo
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph data/test \
                          exp/tri3b/decode_test
    echo
    echo "===== END: tri3b fmllr decode ====="
fi

# if [ $stage -le 7 ]; then
#     echo "===== BEGIN : lmrescore ====="
#     echo
#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang \
#                        data/test exp/tri3b/decode_test
#     echo
#     echo "===== END: lmrescore ====="
# fi

# if [ $stage -le 8 ]; then
#     echo "===== BEGIN : lmrescore_const_arpa ====="
#     echo
#     steps/lmrescore_const_arpa.sh \
#       --cmd "$decode_cmd" data/lang \
#       data/test exp/tri3b/decode_test
#     echo
#     echo "===== END: lmrescore_const_arpa ====="
# fi


# if [ $stage -le 22 ]; then
#     echo "===== BEGIN : DNN training ====="
#     echo
#       local/chain2/run_tdnn_copy.sh --stage $stage

#     echo
#     echo "===== END: DNN training ====="
# fi

end=$(date +%s.%N)    
runtime=$(python -c "print(${end} - ${start})")

echo "Runtime was $runtime"

exit 0
