#/!/bin/bash

games_list=('RoadRunner' 'Asteroids')

for game in "${games_list[@]}"
do
  cd ./autoencoders
  python train_model.py --env "$game""NoFrameskip-v4" --device 2 &
  cd ..

  sleep 1

  cd ./image_completion
  python train_model.py --env "$game""NoFrameskip-v4" --device 2 &
  cd ..

  wait
done