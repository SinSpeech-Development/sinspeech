#!/usr/bin/env bash

ln -s ../../wsj/s5/steps .
ln -s ../../wsj/s5/utils .
ln -s ../../../src .

# data_url='https://drive.google.com/file/d/1feWI1tziCbd336h4Fx2uwGxF8Mj-eVnX/view?usp=sharing'

# wget --no-check-certificate $data_url -O data.zip

# unzip data.zip -d 
# rm -r data.zip

# scp -r /home/lakshan/Documents/fyp/scripts/spliter/V1/data.zip root@157.230.181.156:/root/kaldi/egs/sinspeech/s1