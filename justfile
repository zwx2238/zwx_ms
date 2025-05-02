default:
    @just --list

# 清理所有测试数据
clean:
    @echo "清理所有测试数据..."
    rm -rf test_case*

# 运行所有测试用例
test_all: clean
    @echo "运行所有测试用例..."
    uv run test_precision.py

# 查看测试结果
show_results:
    @echo "测试用例列表:"
    @dir /b /ad test_cases | sort /r

# 比较两个测试结果
compare TEST1 TEST2:
    @echo "比较测试结果: {{TEST1}} vs {{TEST2}}"
    @echo "测试用例1 ({{TEST1}}):"
    @type test_cases\{{TEST1}}\error_config.json
    @echo ""
    @echo "测试用例2 ({{TEST2}}):"
    @type test_cases\{{TEST2}}\error_config.json
    @echo ""
    @echo "整体差异比较:"
    @type test_cases\{{TEST1}}\overall_diff.json
    @echo ""
    @type test_cases\{{TEST2}}\overall_diff.json

# 查看单个测试的问题算子
check_ops TEST:
    @echo "检查测试用例 {{TEST}} 的问题算子:"
    @if exist test_cases\{{TEST}}\analysis\problematic_ops.json (type test_cases\{{TEST}}\analysis\problematic_ops.json) else (echo "未找到问题算子信息")

# 可视化比较
vis_diff TEST:
    @echo "可视化测试用例 {{TEST}} 的差异结果:"
    @echo "打开分析文件夹..."
    @explorer test_cases\{{TEST}}\analysis

# 将项目上传到 GitHub
github-push USERNAME REPO="zwx_ms":
    #!/bin/bash
    # 初始化 git 仓库
    git init
    # 添加 .gitignore 文件 (如果需要的话)
    git add .
    # 创建初始提交
    git commit -m "初始提交: PyTorch 和 MindSpore 框架精度对比工具"
    # 创建远程仓库链接
    git remote add origin https://github.com/{{USERNAME}}/{{REPO}}.git
    # 设置分支名称并推送
    git branch -M main
    git push -u origin main
    # 输出成功信息
    echo "项目已成功上传到 GitHub 仓库: https://github.com/{{USERNAME}}/{{REPO}}"
    echo "请确保您已在 GitHub.com 上手动创建了同名仓库"

# 更新现有的 GitHub 仓库
github-update MESSAGE="更新: 代码改进与功能修复":
    # 添加所有更改
    git add .
    # 提交更改
    git commit -m "{{MESSAGE}}"
    # 推送到远程仓库
    git push
    echo "已成功更新 GitHub 仓库"

