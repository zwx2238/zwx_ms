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

commit:
    git add .
    git commit -m "update"
    http_proxy=127.0.0.1:10792 https_proxy=127.0.0.1:10792 git push origin main

