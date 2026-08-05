PYTEST ?= python -m pytest
TESTS ?= tests
# Extra pytest args injected into every test target. `test-*-no-snowflake`
# targets set this to deselect the Snowflake suites, which need a live
# Snowflake session. GNU make applies a target-specific variable to the
# target's prerequisites too, which is what lets the aggregate targets reuse
# the individual ones.
PYTEST_ARGS ?=

.PHONY: sphinx ghpages \
	test test-ansi test-connect test-connect-regression \
	test-cov test-all test-all-no-snowflake test-no-snowflake

# Default suite: classic Spark session, ANSI mode off.
test:
	$(PYTEST) $(PYTEST_ARGS) $(TESTS)

# Same suite with spark.sql.ansi.enabled=true.
test-ansi:
	$(PYTEST) -c pytest-ansi.ini $(PYTEST_ARGS) $(TESTS)

# The existing Spark suite run against a Spark Connect session. Only the Spark
# tests: the rest are backend-agnostic and gain nothing from a second run.
test-connect:
	$(PYTEST) -c pytest-connect.ini $(PYTEST_ARGS) $(TESTS)/test_spark.py $(TESTS)/comparator

# Spark Connect regression suite. Excluded from every other target by
# `addopts`, and must be its own pytest process: starting a local Connect
# server sets SPARK_LOCAL_REMOTE, after which every later
# SparkSession.builder.getOrCreate() in the process returns the Connect session.
test-connect-regression:
	$(PYTEST) -m spark_connect $(PYTEST_ARGS) $(TESTS)/test_spark_connect.py

test-cov:
	$(PYTEST) $(PYTEST_ARGS) --cov=datacompy --cov-report=term-missing $(TESTS)

# Everything. Needs Java 17, pyspark[connect], and a live Snowflake session.
test-all: test test-ansi test-connect test-connect-regression

# Everything except the Snowflake suites, which error without a live session.
# `--snowflake-session local` is not an alternative here: Snowpark's local
# testing mode is an emulator and most of the Snowflake suite fails against it.
test-no-snowflake: PYTEST_ARGS += -k "not snowflake"
test-no-snowflake: test

test-all-no-snowflake: PYTEST_ARGS += -k "not snowflake"
test-all-no-snowflake: test-all

sphinx:
	cd docs && \
	make -f Makefile clean && \
	make -f Makefile html && \
	cd ..

ghpages:
	git checkout gh-pages && \
	cp -r docs/build/html/* . && \
	git add -u && \
	git add -A && \
	PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "Updated generated Sphinx documentation"
