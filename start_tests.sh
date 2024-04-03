#/!/bin/bash

extractors_list=("lin_concat_ext" "cnn_concat_ext" "combine_ext" "skills_attention_ext" "reservoir_concat_ext" "channels_attention_ext" "fixed_lin_concat_ext")
games_list=("Pong" "Breakout")

for game in "${games_list[@]}"
do
  for extractor in "${extractors_list[@]}"
  do
    echo "Testing $game - $extractor"
    python train_agent.py --env "$game" --device 3 --use-skill True --debug False --extractor "$extractor" --pi 512 --vf 512 &
    sleep 1

  done
  wait
done
echo "All tests finished"