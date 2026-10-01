# Makefile -- moteur HPC (Collatz / Riemann)
CC       ?= gcc
CFLAGS   += -O3 -march=native -fopenmp -Wall -Wextra
LDLIBS   += -lm
BIN      := millennium_engine
PYTHON   ?= python3

.PHONY: all clean test help

all: $(BIN)

$(BIN): millennium_engine.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: $(BIN)
	./$(BIN) collatz-point 27 9780657630
	./$(BIN) collatz-verify 100000 > /dev/null
	$(PYTHON) rh_engine.py frontier > /dev/null
	$(PYTHON) rh_engine.py robin 1e12 > /dev/null
	@echo "OK : tous les tests passent."

clean:
	rm -f $(BIN) results/*.csv results/*.json
	find . -name __pycache__ -type d -exec rm -rf {} +

help:
	@echo "cibles : all (defaut) | test | clean"
	@echo "binaires : ./$(BIN) <mode> ...   (voir ./$(BIN) sans arguments)"
	@echo "python   : $(PYTHON) rh_engine.py <sous-module> ..."
