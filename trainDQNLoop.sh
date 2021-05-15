#!/bin/bash

timestamp=$(date +%s)
frames=250000 # 250000 is about 37 minutes msi
lr=0.0001 # 0.0001
algo=dqn

save_interval=1 # 1 , if save_interval == 1, will save after each train/test run
log_interval=1000 # 5000
testeveryNlogs=1 # 5

eps_dec=40000 # 70000
conv=0 # 0
gamma=0.99

phonesAmount=1
dirtAmount=1

numKnownConfigs=1
numUnknownConfigPhones=1
numTestConfigs=7

totalTrainingEpisodes=$(($numKnownConfigs+$numUnknownConfigPhones))

modelDirectory="${timestamp}_${algo}__pa${phonesAmount}_da${dirtAmount}_nkc${numKnownConfigs}_nucp${numUnknownConfigPhones}_ntc${numTestConfigs}_F${frames}_LR${lr}_G${gamma}_E${eps_dec}"

echo "saving all runs to folder: '" . $modelDirectory

for(( i = 1; i<=10; i++))
do
  python -m scripts.trainDQN --conv $conv --seed $i --visualise 0 --env MiniGrid-ScalableOversightTRAINING-v0  --testenv MiniGrid-ScalableOversightTEST-v0  --testeveryNlogs $testeveryNlogs  --save_interval $save_interval --log_interval $log_interval --frames $frames --testepisodes $numTestConfigs --trainingepisodes $totalTrainingEpisodes --dir $modelDirectory --lr $lr --eps_dec $eps_dec --gamma $gamma
done

python -m scripts.postProcessFiles --dir $modelDirectory --showGraph 0 --storagePath .
