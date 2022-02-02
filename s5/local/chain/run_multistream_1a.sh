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
affix=1a
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

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # MFCC to filterbank
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  
  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct
  
  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256

  relu-batchnorm-dropout-layer name=tdnn6a $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf8a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf9a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf10a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf11a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf12a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf13a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf14a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf15a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf16a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf17a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf18a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf19a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf20a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf21a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf22a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf23a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  
  relu-batchnorm-dropout-layer name=tdnn6b $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf8b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf9b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf10b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf11b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf12b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf13b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf14b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf15b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf16b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf17b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf18b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf19b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf20b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf21b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf22b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf23b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9

  relu-batchnorm-dropout-layer name=tdnn6c $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf8c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf9c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf10c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf11c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf12c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf13c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf14c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf15c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf16c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf17c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf18c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf19c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf20c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf21c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf22c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf23c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  
  relu-batchnorm-dropout-layer name=tdnn17 $affine_opts input=Append(tdnnf23a,tdnnf23b,tdnnf23c) dim=768
  linear-component name=prefinal-l dim=256 $linear_opts
  
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
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
    --trainer.num-chunk-per-minibatch 16 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.00015 \
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
  for data in test; do
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