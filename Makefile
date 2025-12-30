run:
	mlx-server launch \
	--model-path mlx-community/Qwen3-1.7B-4bit \
	--model-type lm \
	--max-concurrency 1 \
	--queue-timeout 300 \
	--queue-size 100

install:
	pip install -e .
