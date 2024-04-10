#/!/bin/bash

games_list=('MsPacman' 'Seaquest' 'Qbert' 'Asteroids' 'Enduro' 'SpaceInvaders' 'BeamRider')

for game in "${games_list[@]}"
do
  cd ./autoencoders
  python train_model.py --env "$game""NoFrameskip-v4" --device 1 &
  cd ..

  sleep 1

  cd ./image_completion
  python train_model.py --env "$game""NoFrameskip-v4" --device 1 &
  cd ..

  wait
done