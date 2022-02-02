#!/usr/bin/env bash

stage=0
LOG_LOCATION=`pwd`/../logs

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"run_mfcc_"$stage.log 2>&1

nj=$(nproc)

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
    # Making spk2utt files
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
    
    utils/validate_data_dir.sh data/train --no-feats
    utils/fix_data_dir.sh data/train

    utils/validate_data_dir.sh data/test --no-feats
    utils/fix_data_dir.sh data/test
fi

if [ $stage -le 1 ]; then

    mfccdir=mfcc

    echo "===== BEGIN : Train Set MFCC feature extraction ====="
    echo
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
    steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
    echo
    echo "===== END: Train Set MFCC feature extraction ====="

    echo "===== BEGIN : Test Set MFCC feature extraction ====="    
    echo
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test $mfccdir
    steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir
    echo
    echo "===== END: Test Set MFCC feature extraction ====="

fi

exit 0