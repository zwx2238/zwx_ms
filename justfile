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

commit message="update":
    uv run ruff format .
    git add .
    git commit -m {{message}}
    http_proxy=127.0.0.1:10792 https_proxy=127.0.0.1:10792 git push -u origin master

