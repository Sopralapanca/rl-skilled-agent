#/!/bin/bash
set -e

##games_list=('Asteroids' 'Ms_Pacman' 'Space_Invaders' 'Seaquest' 'Qbert' 'Pong' 'Breakout')
##games_list=('Seaquest' 'Qbert' 'Pong' 'Breakout')
#games_list=('Asteroids')
##extractors_list=('lin_concat_ext' 'fixed_lin_concat_ext' 'cnn_concat_ext' 'combine_ext' 'dotproduct_attention_ext' 'wsharing_attention_ext' 'reservoir_concat_ext')
#extractors_list=('lin_concat_ext' 'fixed_lin_concat_ext' 'cnn_concat_ext' 'combine_ext' 'dotproduct_attention_ext' 'wsharing_attention_ext')
#
#fd=(256 512 1024)
#ro=(512 1024)
#cv=(1 2 3)
#
#for game in "${games_list[@]}"
#do
#  #python train_agent.py --env "$game" --device 2 --use-skill False --debug False &
#
#  for extractor in "${extractors_list[@]}"
#  do
#    if [ "$extractor" = "fixed_lin_concat_ext" ] || [ "$extractor" = "dotproduct_attention_ext" ] || [ "$extractor" = "wsharing_attention_ext" ] ; then
#
#      for d in "${fd[@]}"
#      do
#        python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" --fd "$d" &
#      done
#      wait
#
#    elif [ "$extractor" = "cnn_concat_ext" ] ; then
#      for l in "${cv[@]}"
#      do
#        python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" --cv "$l" &
#      done
#
#    elif [ "$extractor" = "reservoir_concat_ext" ] ; then
#      for s in "${ro[@]}"
#      do
#        python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" --ro "$s" &
#      done
#
#    else
#      python train_agent.py --env "$game" --device 2 --use-skill True --debug False --extractor "$extractor" &
#    fi
#
#  done
#done


python evaluate_agents.py --env Ms_Pacman --device 2
wait
python evaluate_agents.py --env Pong --device 2
