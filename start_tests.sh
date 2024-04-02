#/!/bin/bash

python train_agent.py --env Breakout --device 3 --use-skill True --debug False --extractor lin_concat_ext &
sleep 1
python train_agent.py --env Pong --device 3 --use-skill True --debug False --extractor lin_concat_ext &

wait
echo "All tests finished"