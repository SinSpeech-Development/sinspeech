#!/usr/bin/env bash
start=$(date +%s.%N)

stage=5
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

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 50000 data/train_50000short

fi

start1=$(date +%s.%N)
# train a monophone system
if [ $stage -le 1 ]; then
    echo "===== BEGIN : Train 50, 000 Short Mono ====="
    echo
    # TODO(galv): Is this too many jobs for a smaller dataset?
    steps/train_mono.sh --boost-silence 1.25 --nj $nj \
    --totgauss 2000 --cmd "$train_cmd" \
        data/train_50000short data/lang exp/mono
    echo
    echo "===== END: Train 50, 000 Short Mono ====="

    echo "===== BEGIN : mono align ====="
    echo
    steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/mono exp/mono_ali_train
    echo
    echo "===== END: mono align ====="
fi
end1=$(date +%s.%N)    

start2=$(date +%s.%N)
# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 2 ]; then
    echo "===== BEGIN : Train delta + delta-delta triphone ====="
    echo
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
        2000 50000 \
        data/train data/lang exp/mono_ali_train exp/tri1
    echo
    echo "===== END: Train delta + delta-delta triphone ====="

    echo "===== BEGIN : tri1 align ====="
    echo
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri1 exp/tri1_ali_train
    echo
    echo "===== END: tri1 align ====="
fi
end2=$(date +%s.%N)    

start3=$(date +%s.%N)
# train an LDA+MLLT system.
if [ $stage -le 3 ]; then
    echo "===== BEGIN: train LDA+MLLT - tri2b model ====="
    echo
    steps/train_lda_mllt.sh  --cmd "$train_cmd" \
        --splice-opts "--left-context=4 --right-context=4" \
        2500 100000 \
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
end3=$(date +%s.%N)    

start4=$(date +%s.%N)
# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 4 ]; then
    echo "===== BEGIN : train LDA+MLLT+SAT - tri3b model ====="
    echo
    steps/train_sat.sh --cmd "$train_cmd" \
        4500 105000 \
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
end4=$(date +%s.%N)    


if [ $stage -le 5 ]; then
    echo "===== BEGIN : DNN training ====="
    echo
      local/chain/tdnn_f.sh

    echo
    echo "===== END: DNN training ====="
fi

end=$(date +%s.%N)  

runtime1=$(python -c "print(${end1} - ${start1})")
runtime2=$(python -c "print(${end2} - ${start2})")
runtime3=$(python -c "print(${end3} - ${start3})")
runtime4=$(python -c "print(${end4} - ${start4})")

runtime=$(python -c "print(${end} - ${start})")

echo "Total run time for mono was $runtime1"
echo "Total run time for tri1 was $runtime2"
echo "Total run time for tri2 was $runtime3"
echo "Total run time for tri3 was $runtime4"


echo "Total run time was $runtime"

exit 0
