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