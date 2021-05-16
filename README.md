With thanks to Lucas Willems (lcswillems), original author of https://github.com/lcswillems/rl-starter-files

This is designed to be used with https://github.com/louism33/gym-minigrid to train agents on the Scalable Supervision gridworld. This was for my [thesis](Thesis.pdf).

Easiest is to run:
`
./trainPPOLoop.sh
`
of 
`
./trainDQNLoop.sh
`
Make sure to change the underlying gridworld hyperparameters in https://github.com/louism33/gym-minigrid/blob/master/gym_minigrid/envs/scalableoversight.py.

You will need to use this in conjunction with https://github.com/louism33/torch-ac and https://github.com/louism33/gym-minigrid

