default:
    @just --list

# 清理所有测试数据
clean:
    @echo "清理所有测试数据..."
    rm -rf test_case* rank_*

# 运行所有测试用例
test_all: clean
    @echo "运行所有测试用例..."
    # uv run tests/test_shared_weights.py
    uv run tests/test_precision.py --parallel

# 运行所有测试用例
test:
    pytest tests/test_precision_cases.py -v

# 运行特定的测试用例
test-case CASE:
    pytest tests/test_precision_cases.py::{{CASE}} -v

# 运行命令行工具
cli *ARGS:
    python precision_compare/cli/precision_test.py {{ARGS}}

# 运行并行测试示例
test-parallel:
    python precision_compare/cli/precision_test.py --parallel

# 运行单个错误类型测试示例
test-single ERROR_TYPE='weight_noise' MODULE_PATH='model.layers.1.input_layernorm' FRAMEWORK='mindspore': clean
    python precision_compare/cli/precision_test.py --error-type {{ERROR_TYPE}} --module-path {{MODULE_PATH}} --framework {{FRAMEWORK}}

commit message="update":
    uv run ruff format .
    git add .
    git commit -m {{message}}
    http_proxy=127.0.0.1:10792 https_proxy=127.0.0.1:10792 git push -u origin master

