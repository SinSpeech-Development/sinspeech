#!/usr/bin/env bash

# Copyright 2021 ASAPP (author: Kyu J. Han)
# MIT License

# This recipe is based on a paper titled "Multistream CNN for Robust Acoustic Modeling",
# https://arxiv.org/abs/2005.10470.

set -euo pipefail

# configs for 'chain'
stage=0
decode_nj=$(nproc)
train_set=train
gmm=tri3b
nnet3_affix=

# The rest are config specific to this script. Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=17b
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# tdnn options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# if true, it will run the last decoding section
test_online_decoding=true

# end of the configurations section

echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

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
# this directory change depending on the recipie
dir=exp/chain${nnet3_affix}/multistream_cnn${affix:+_$affix}_sp


if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=512
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=512 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=512 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 15 ]; then
  echo "training started"
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 512 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.0005 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
  echo "training finished"

fi

graph_dir=$dir/graph
if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang $dir $graph_dir
  # remove <UNK> from the graph, and convert back to const-FST.
  # fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
  #   fstconvert --fst_type=const > $graph_dir/temp.fst
  # mv $graph_dir/temp.fst $graph_dir/HCLG.fst
fi

if [ $stage -le 17 ]; then
  echo "decoding started"
  frames_per_chunk=$(echo $frames_per_eg | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true
  for data in test valid; do
      (
    steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $graph_dir data/${data}_hires ${dir}/decode_${data} || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
  echo "decoding finished"
fi
exit 0;
