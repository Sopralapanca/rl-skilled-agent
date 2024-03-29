#/!/bin/bash

python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor lin_concat_ext &
sleep 1
python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor cnn_concat_ext &
sleep 1
python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor combine_ext &
sleep 1
python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor self_attention_ext &
sleep 1
python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor reservoir_concat_ext &


wait
echo "All tests finished"


