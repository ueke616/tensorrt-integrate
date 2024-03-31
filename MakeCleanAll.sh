#!/bin/bash

# 当前目录下递归深度1的所有目录
for d in */ ; do
    # 检查是否存在Makefile
    if [ -f "$d/Makefile" ]; then
        echo "Running make clean in $d"
        # 进入目录
        cd "$d"
        # 执行make clean命令
        make clean
        # 返回到上一级目录
        cd ..
    fi
done
