#!/bin/bash

# スクリプトのエラーで処理を停止
set -e

# docker-compose.yml はこのスクリプトと同じディレクトリにある
COMPOSE_FILE="docker/docker-compose.yml"

# ヘルプメッセージを表示する関数
show_help() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "This script must be run from within the 'docker' directory."
    echo ""
    echo "Commands:"
    echo "  build       Build or rebuild the docker image."
    echo "  shell       Start an interactive shell in the container."
    echo "  up          Start the development container in the background."
    echo "  down        Stop and remove the container."
    echo "  run [args]  Execute a command in a new container (e.g., ./run.sh run python demo/ball.py)."
    echo "  test [path] Run pytest (e.g., ./run.sh test tests/)."
    echo "  logs        Follow logs from the running container."
    echo "  clean       Remove dangling docker images and volumes."
    echo ""
}

# メインの処理
case "$1" in
    build)
        echo "🚀 Building docker image..."
        docker-compose -f "$COMPOSE_FILE" build
        ;;
    shell)
        echo "💻 Starting interactive shell..."
        docker-compose -f "$COMPOSE_FILE" run --rm tennis-dev bash
        ;;
    up)
        echo "⬆️  Starting container in background..."
        docker-compose -f "$COMPOSE_FILE" up -d tennis-dev
        ;;
    down)
        echo "⬇️  Stopping container..."
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    run)
        shift # "run" を引数リストから削除
        if [ -z "$1" ]; then
            echo "Error: Please specify a command to run."
            echo "Example: ./run.sh run python demo/ball.py"
            exit 1
        fi
        echo "🏃 Running command: $@"
        docker-compose -f "$COMPOSE_FILE" run --rm tennis-dev "$@"
        ;;
    test)
        shift # "test" を引数リストから削除
        TEST_PATH=${1:-"tests/"} # 引数がなければ "tests/" を使用
        echo "🧪 Running tests in: $TEST_PATH"
        docker-compose -f "$COMPOSE_FILE" run --rm tennis-dev python -m pytest "$TEST_PATH"
        ;;
    logs)
        echo "📄 Following logs..."
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    clean)
        echo "🧹 Cleaning up docker system..."
        docker-compose -f "$COMPOSE_FILE" down --volumes
        docker system prune -f
        ;;
    *)
        show_help
        ;;
esac

exit 0