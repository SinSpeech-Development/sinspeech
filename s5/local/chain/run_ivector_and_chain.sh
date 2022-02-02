#!/usr/bin/env bash

set -euo pipefail

stage=0
train_set=train
gmm=tri3b
nnet3_affix=
tree_affix=

echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;


# directory contains the final hmm model
gmm_dir=exp/$gmm

# directory contains the training alignments of speed perturbed data
ali_dir=exp/${gmm}_ali_${train_set}_sp

# does not exist yet, new lang directory with the new topology
lang=data/lang_chain

# data directory
# directory of training set, which are the high-resolution MFCCs
train_data_dir=data/${train_set}_sp_hires
# directory of training set, which are the low-resolution MFCCs
lores_train_data_dir=data/${train_set}_sp

# nnet3 directory
# directory contains i-vectors
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

# chain directory
# does not exist yet, directory to put the new decision tree
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
# does not exist yet, directory to put lattices
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats


# if we are using the speed-perturbed data we need to generate
# alignments for it.
for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
            $train_ivector_dir/ivector_online.scp \
            $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

# Please take this as a reference on how to specify all the options of
# local/chain/run_chain_common.sh
local/chain/run_chain_common.sh --stage $stage \
                                --gmm-dir $gmm_dir \
                                --ali-dir $ali_dir \
                                --lores-train-data-dir ${lores_train_data_dir} \
                                --lang $lang \
                                --lat-dir $lat_dir \
                                --num-leaves 4500 \
                                --tree-dir $tree_dir || exit 1;