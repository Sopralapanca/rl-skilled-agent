#/!/bin/bash

games_list=('Ms_Pacman' 'Space_Invaders' 'Road_Runner' 'Beam_Rider')

for game in "${games_list[@]}"
do
  python create_dataset.py --env "$game" &

  sleep 1
done
