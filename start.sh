#! /bin/bash
sudo apt-get update

printf '%s %s Running startup script \n' $(date +%Y-%m-%d) $(date +%H:%M:%S) >> '{absolute_path}/startup_script.log'

cd ./KneeSegmentation

# git update application
git pull origin master
nbx --inplace KneeTrain.ipynb
nbx --inplace KneeTest.ipynb

# shutdown -h now÷