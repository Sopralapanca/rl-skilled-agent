#/!/bin/bash

cd ./autoencoders
python train_model.py --env BreakoutNoFrameskip-v4 --device 2 &
cd ..

sleep 1

cd ./image_completion
python train_model.py --env BreakoutNoFrameskip-v4 --device 2 &
cd ..

wait
echo "Skills training finished"