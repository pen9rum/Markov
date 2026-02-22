#!/bin/bash
# 批量实验工具 - 从有效组合中抽取并进行LLM分析

# 切换到src目录
cd "$(dirname "$0")/../src" || exit

# 运行批量实验脚本
python -m sys.path.insert(0, '.') -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../tools')
from batch_experiment import main
main()
" || python ../tools/batch_experiment.py
