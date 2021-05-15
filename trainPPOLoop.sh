#!/bin/bash

timestamp=$(date +%s)
frames=3000000
entropyCoef=0.25   # ppo: 0.25, a2c: 0.25
valueLoss=1 # ppo: 1, a2c: 1
discount=0.99 # ppo: 0.99, a2c: 0.99
clip=0.2 # ppo: 0.2, a2c: 0.2
gae=0.95 # ppo: 0.95, a2c: 0.95
maxGradNorm=0.5 # ppo: 0.5, a2c: 0.5
lr=0.0011 # ppo: 0.0011, a2c: 0.001
algo=ppo
allInterval=1 # ppo: 10, a2c: 20

phonesAmount=2
dirtAmount=1

numKnownConfigs=1
numUnknownConfigPhones=1
numTestConfigs=1

totalTrainingEpisodes=$(($numKnownConfigs+$numUnknownConfigPhones))

modelDirectory="${timestamp}_${algo}_pa${phonesAmount}_da${dirtAmount}_nkc${numKnownConfigs}_nucp${numUnknownConfigPhones}_ntc${numTestConfigs}_F${frames}_E${entropyCoef}_V${valueLoss}_D${discount}_C${clip}_LR${lr}_GAE${gae}_MGN${maxGradNorm}"

echo "saving all runs to folder: '" . $modelDirectory

for(( i = 1; i<=10; i++))
do
  python3 -m scripts.trainTest --algo $algo  --env MiniGrid-ScalableOversightTRAINING-v0  --save_interval $allInterval --log_interval $allInterval --frames $frames --seed $i --entropy-coef $entropyCoef --testenv MiniGrid-ScalableOversightTEST-v0 --testeveryNlogs 1 --testepisodes $numTestConfigs --trainingepisodes $totalTrainingEpisodes --value-loss-coef $valueLoss --discount $discount --dir $modelDirectory --clip-eps $clip --gae-lambda $gae --max-grad-norm $maxGradNorm --lr $lr
done

python3 -m scripts.postProcessFiles --dir $modelDirectory --showGraph 0 --storagePath .