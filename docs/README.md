# Documentation

This folder contains the scripts necessary to build documentation. You can
find the documentation at <https://nvidia-merlin.github.io/systems/main>.



1. Follow the instructions to create a Python developer environment in
   the repository README.

2. Install required documentation tools and extensions:

   ```shell
   cd models
   pip install -r requirements-dev.txt
   ```

3. Transform the documentation to HTML output:

   ```shell
   make -C docs clean html
   ```

   This should run Sphinx in your shell, and output HTML in
   `docs/build/html/index.html`

## Preview the documentation build

1. To view the docs build, run the following command:

   ```shell
   python -m http.server -d docs/build/html
   ```

