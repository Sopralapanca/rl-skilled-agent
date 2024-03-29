#/!/bin/bash

python ./autoencoders/train_model.py --env Breakout --device 3 &
sleep 1
python ./image_completion/train_model.py --env Breakout --device 3 &

wait
echo "Skills training finished"


