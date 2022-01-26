#!/usr/bin/env bash
set -euo pipefail


# configs for 'chain'
stage=0
decode_nj=$(nproc)
train_set=train
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1d
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=true  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

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

# does not exist yet, directory to put the new decision tree
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}

# does not exist yet, new lang directory with the new topology
lang=data/lang_chain

# does not exist yet, directory to put lattices
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats

# does not exist yet, directory to put #TODO
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp

# directory of training set, which are the high-resolution MFCCs
train_data_dir=data/${train_set}_sp_hires

# directory of training set, which are the low-resolution MFCCs
lores_train_data_dir=data/${train_set}_sp

# directory contains i-vectors
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

# if we are using the speed-perturbed data we need to generate
# alignments for it.

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
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
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=256
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=256 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=256 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=256 small-dim=192
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
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 6 \
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

if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang $dir $tree_dir/graph
fi


if [ $stage -le 17 ]; then
  echo "decoding started"
  frames_per_chunk=$(echo $frames_per_eg | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true
  for data in test; do
      (
    steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph data/${data}_hires ${dir}/decode_${data} || exit 1
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