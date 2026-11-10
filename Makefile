.PHONY: install test prior train edge
install:
	python -m pip install -e '.[test]'
test:
	pytest -q
prior:
	python scripts/02_generate_prior_and_pretrain.py --tables 100 --steps 0
train:
	python -m nanomaternalpfn.train --steps 100
edge:
	python scripts/04_run_edgebench_profiler.py
