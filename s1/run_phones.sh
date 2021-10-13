#!/usr/bin/env bash

stage=22
LOG_LOCATION=`pwd`/../logs

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

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 500 data/train_500short

fi

# train a monophone system
if [ $stage -le 1 ]; then
    echo "===== BEGIN : Train 500 Short Mono ====="
    echo
    # TODO(galv): Is this too many jobs for a smaller dataset?
    steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/mono
    echo
    echo "===== END: Train 500 Short Mono ====="

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

    # decode using the tri3b model
    echo "===== BEGIN : make graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri3b exp/tri3b/graph
    echo
    echo "===== END: make graph ====="
fi

if [ $stage -le 6 ]; then

    echo "===== BEGIN : fmllr decode ====="
    echo
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph data/test \
                          exp/tri3b/decode_test
    echo
    echo "===== END: fmllr decode ====="
fi

if [ $stage -le 7 ]; then
    echo "===== BEGIN : lmrescore ====="
    echo
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang \
                       data/test exp/tri3b/decode_test
    echo
    echo "===== END: lmrescore ====="
fi

if [ $stage -le 8 ]; then
    echo "===== BEGIN : lmrescore_const_arpa ====="
    echo
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang \
      data/test exp/tri3b/decode_test
    echo
    echo "===== END: lmrescore_const_arpa ====="
fi


if [ $stage -le 22 ]; then
    echo "===== BEGIN : DNN training ====="
    echo
      local/chain2/run_tdnn_copy.sh --stage $stage

    echo
    echo "===== END: DNN training ====="
fi

exit 0
