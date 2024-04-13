#/!/bin/bash

#extractors_list=("fixed_lin_concat_ext" "cnn_concat_ext" "combine_ext" "reservoir_concat_ext" "fixed_lin_concat_ext")
#games_list=("Pong")

#for game in "${games_list[@]}"
#do
#  for extractor in "${extractors_list[@]}"
#  do
#    python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" --fd 256 &
#    sleep 1

#  done
#  wait

#done
#echo "All tests finished"


python train_agent.py --env "Breakout" --device 2 --use-skill True --debug False --extractor "fixed_lin_concat_ext" --fd 256 &
sleep 1

python train_agent.py --env "Breakout" --device 2 --use-skill True --debug False --extractor "fixed_lin_concat_ext" --fd 512 &
sleep 1

python train_agent.py --env "Breakout" --device 2 --use-skill True --debug False --extractor "fixed_lin_concat_ext" --fd 1024 &
sleep 1

python train_agent.py --env "Breakout" --device 2 --use-skill True --debug False --extractor "fixed_lin_concat_ext" --fd 2048 &
sleep 1


python train_agent.py --env "Pong" --device 3 --use-skill True --debug False --extractor "cnn_concat_ext" --cv 2 &
sleep 1

python train_agent.py --env "Pong" --device 3 --use-skill True --debug False --extractor "cnn_concat_ext" --cv 3 &
sleep 1

python train_agent.py --env "Pong" --device 3 --use-skill True --debug False --extractor "reservoir_concat_ext" --ro 512 &
sleep 1

python train_agent.py --env "Pong" --device 3 --use-skill True --debug False --extractor "reservoir_concat_ext" --ro 1024 &
sleep 1

python train_agent.py --env "Pong" --device 3 --use-skill True --debug False --extractor "reservoir_concat_ext" --fd 2048 &
sleep 1




python train_agent.py --env "Pong" --device 0 --use-skill True --debug False --extractor "self_attention_ext" --fd 256 --heads 4 &
sleep 1

python train_agent.py --env "Pong" --device 0 --use-skill True --debug False --extractor "self_attention_ext" --fd 512 --heads 4 &
sleep 1

python train_agent.py --env "Pong" --device 0 --use-skill True --debug False --extractor "self_attention_ext" --fd 1024 --heads 4 &
sleep 1

python train_agent.py --env "Pong" --device 0 --use-skill True --debug False --extractor "self_attention_ext" --fd 2048 --heads 4 &
sleep 1



python train_agent.py --env "Pong" --device 1 --use-skill True --debug False --extractor "dotproduct_attention_ext" --fd 256 &
sleep 1

python train_agent.py --env "Pong" --device 1 --use-skill True --debug False --extractor "dotproduct_attention_ext" --fd 512 &
sleep 1

python train_agent.py --env "Pong" --device 1 --use-skill True --debug False --extractor "dotproduct_attention_ext" --fd 1024 &
sleep 1

python train_agent.py --env "Pong" --device 1 --use-skill True --debug False --extractor "dotproduct_attention_ext" --fd 2048 &
sleep 1

