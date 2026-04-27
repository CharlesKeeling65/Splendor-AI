#!/bin/bash

set -e

cd /home/wangyb/TestProject/Splendor-AI

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 未安装，请先安装 uv"
  exit 1
fi

uv sync

echo "完成。请执行："
echo "source .venv/bin/activate"
echo "然后运行："
echo "splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t"
