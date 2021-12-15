#!/usr/bin/env bash

start=$(date +%s.%N)

stage=0
LOG_LOCATION=`pwd`/logs

model=$1

# hyper param for mono
hp_mono_totgauss=1000

# hyper param for tri1
hp_tri1_numleaves=2000
hp_tri1_totgauss=10000

# hyper param for tri2
hp_tri2_left_context=3
hp_tri2_right_context=3
hp_tri2_numleaves=2500
hp_tri2_totgauss=15000

# hyper param for tri3
hp_tri3_numleaves=2500
hp_tri3_totgauss=15000

splice_opts="--left-context=""$hp_tri2_left_context"" --right-context=""$hp_tri2_right_context"

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"hyper_param_tuning_"$model.log 2>&1

if [ "$model" = "mono" ]; then
    echo "=== Hyper params for Mono Model ==="
    echo "totgauss: "$hp_mono_totgauss
    echo
fi

if [ "$model" = "tri1" ]; then
    echo "=== Hyper params for Tri1 Model ==="
    echo "number of leaves: "$hp_tri1_numleaves
    echo "totgauss: "$hp_tri1_totgauss
    echo
fi

if [ "$model" = "tri2" ]; then
    echo "=== Hyper params for Tri2 Model ==="
    echo "left_context: "$hp_tri2_left_context
    echo "right_context: "$hp_tri2_right_context
    echo "number of leaves: "$hp_tri2_numleaves
    echo "totgauss: "$hp_tri2_totgauss
    echo 
fi

if [ "$model" = "tri3" ]; then
    echo "=== Hyper params for Tri3 Model ==="
    echo "number of leaves: "$hp_tri3_numleaves
    echo "totgauss: "$hp_tri3_totgauss
    echo 
fi

nj=$(nproc)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [[ $stage -le 0 && "$model" == "mono" ]]; then

  # Get the shortest 5000 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 50000 data/train_50000short

fi

# train a monophone system
if [[ $stage -le 1 && "$model" == "mono" ]]; then

    echo "===== BEGIN : Train 50000 Short Mono ====="
    echo
    # TODO(galv): Is this too many jobs for a smaller dataset?
    steps/train_mono.sh --boost-silence 1.25 --nj $nj \
                        --totgauss $hp_mono_totgauss --cmd "$train_cmd" \
                        data/train_50000short data/lang exp/mono
    echo
    echo "===== END: Train 50000 Short Mono ====="

    # decode using the mono model
    echo "===== BEGIN : make mono graph ====="
    echo
    utils/mkgraph.sh data/lang \
                    exp/mono exp/mono/graph
    echo
    echo "===== END: make mono graph ====="

    echo "===== BEGIN : mono decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/mono/graph data/valid \
                          exp/mono/decode_valid
    echo
    echo "===== END: mono decode ====="

fi

# train a first delta + delta-delta triphone system on all utterances
if [[ $stage -le 2 && "$model" == "tri1" ]]; then

    echo "===== BEGIN : mono align ====="
    echo
    # steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        # data/train data/lang exp/mono exp/mono_ali_train
    echo
    echo "===== END: mono align ====="
    
    echo "===== BEGIN : Train delta + delta-delta triphone ====="
    echo
    # steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
        # $hp_tri1_numleaves $hp_tri1_totgauss data/train data/lang \
        # exp/mono_ali_train exp/tri1
    echo
    echo "===== END: Train delta + delta-delta triphone ====="

    # decode using the tri1 model
    echo "===== BEGIN : make tri1 graph ====="
    echo
    # utils/mkgraph.sh data/lang \
                #    exp/tri1 exp/tri1/graph
    echo
    echo "===== END: make tri1 graph ====="

    echo "===== BEGIN : tri1 decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                            --stage 2 \
                          exp/tri1/graph data/valid \
                          exp/tri1/decode_valid
    echo
    echo "===== END: tri1 decode ====="
    
fi

# train an LDA+MLLT system.
if [[ $stage -le 3 && "$model" == "tri2" ]]; then

    echo "===== BEGIN : tri1 align ====="
    echo
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri1 exp/tri1_ali_train
    echo
    echo "===== END: tri1 align ====="

    echo "===== BEGIN: train LDA+MLLT - tri2b model ====="
    echo
    steps/train_lda_mllt.sh  --cmd "$train_cmd" \
        --splice-opts $splice_opts \
        $hp_tri2_numleaves $hp_tri2_totgauss \
        data/train data/lang exp/tri1_ali_train exp/tri2b
    echo
    echo "===== END: train LDA+MLLT - tri2b model ====="

    # decode using the tri2b model
    echo "===== BEGIN : make tri2b graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri2b exp/tri2b/graph
    echo
    echo "===== END: make tri2b graph ====="

    echo "===== BEGIN : tri2b decode ====="
    echo
    steps/decode.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri2b/graph data/valid \
                          exp/tri2b/decode_valid
    echo
    echo "===== END: tri2b decode ====="
fi

# Train tri3b, which is LDA+MLLT+SAT
if [[ $stage -le 4 && "$model" == "tri3" ]]; then
    echo "===== BEGIN: tri2b model align ====="
    echo
    # Align utts using the tri2b model
    steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
        data/train data/lang exp/tri2b exp/tri2b_ali_train
    echo
    echo "===== END: tri2b model align ====="

    echo "===== BEGIN : train LDA+MLLT+SAT - tri3b model ====="
    echo
    steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
        data/train data/lang exp/tri2b_ali_train exp/tri3b
    echo
    echo "===== END: train LDA+MLLT+SAT - tri3b model ====="

    # decode using the tri3b model
    echo "===== BEGIN : make tri3b graph ====="
    echo
    utils/mkgraph.sh data/lang \
                   exp/tri3b exp/tri3b/graph
    echo
    echo "===== END: make tri3b graph ====="

    echo "===== BEGIN : tri3b fmllr decode ====="
    echo
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph data/valid \
                          exp/tri3b/decode_valid
    echo
    echo "===== END: tri3b fmllr decode ====="
fi

end=$(date +%s.%N)    
runtime=$(python -c "print(${end} - ${start})")

echo "Runtime was $runtime"

exit 0