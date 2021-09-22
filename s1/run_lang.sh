#!/usr/bin/env bash

stage=0
lm_order=3
LOG_LOCATION=`pwd`/../logs

if [ ! -d "$LOG_LOCATION" ]; then
  mkdir -p $LOG_LOCATION
fi

# log the terminal outputs
exec >> $LOG_LOCATION/"run_lang_"$stage.log 2>&1

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
    # save the tempory results in data/local/lang
    # save the final lang data in data/lang
    echo "===== BEGIN : PREPARE L.fst ====="
    echo
    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
    echo
    echo "===== END: PREPARE L.fst ====="
fi


if [ $stage -le 1 ]; then

    # setting up the SRILM TOOL path to PATH
    loc='which ngram-count';
    if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
            sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
            sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
            export PATH=$PATH:$sdir
        else
            echo "SRILM toolkit is probably not installed.
                        Instructions: tools/install_srilm.sh"
            exit 1
        fi
    fi

    # count n-grams in corpus.txt file
    # save the results in .arpa.gz file
    local=data/local    
    mkdir $local/tmp

    echo "===== BEGIN : PREPARE lm.arpa.gz ====="
    echo
    ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt \ 
            -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa.gz
    echo
    echo "===== END: PREPARE lm.arpa.gz ====="

fi

if [ $stage -le 2 ]; then

    tmpdir=data/local/lm_tmp.$$

    # remove the temporary directory upon exiting
    trap "rm -r $tmpdir" EXIT
    mkdir -p $tmpdir

    local=data/local
    lang=data/lang

    echo "===== BEGIN : PREPARE G.fst ====="
    echo
    gunzip -c $local/tmp/lm.arpa.gz | \
        arpa2fst --disambig-symbol=#0 \
        --read-symbol-table=$lang/words.txt - $lang/G.fst
    echo
    echo "===== END: G.fst ====="

    utils/validate_lang.pl --skip-determinization-check $lang || exit 1;

fi

exit 0