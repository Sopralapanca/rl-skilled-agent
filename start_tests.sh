#/!/bin/bash

extractors_list=("lin_concat_ext" "cnn_concat_ext" "combine_ext" "reservoir_concat_ext" "fixed_lin_concat_ext")
games_list=("Pong" "Breakout")
layers=(256 512 1024)

for game in "${games_list[@]}"
do
  for l in "${layers[@]}"
  do
    python train_agent.py --env "$game" --device 2 --use-skill False --debug False --pi "$l" --vf "$l" &
    echo "Testing $game - No Skill - net arch size $l"
    for extractor in "${extractors_list[@]}"
    do
      echo "Testing $game - $extractor - net arch size $l"
      python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" --pi "$l" --vf "$l" &
      sleep 1

    done
    wait
  done
done
echo "All tests finished"