#/!/bin/bash
set -e
games_list=('Pong')
extractors_list=('lin_concat_ext' 'fixed_lin_concat_ext' 'cnn_concat_ext' 'combine_ext' 'dotproduct_attention_ext' 'wsharing_attention_ext' 'reservoir_concat_ext')

for extractor in "${extractors_list[@]}"
do
  for game in "${games_list[@]}"
  do
    echo "Testing $game with extractor $extractor"
    python train_agent.py --env "$game" --use-skill True --debug True --extractor "$extractor"
    echo "Done"
  done
done
