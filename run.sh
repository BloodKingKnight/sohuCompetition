#!/bin/bash
nohup python run.py > logs/debert_base.log 2>&1 &
tail -f logs/debert_base.log
sleep 300
sh /mistgpu/shutdown.sh
