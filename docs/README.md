# Documentation

This folder contains the scripts necessary to build documentation for the
Merlin Systems library. You can find the [generated documentation
here](https://nvidia-merlin.github.io/systems).



1. Follow the instructions to create a Python developer environment. See the
   [installation instructions](https://github.com/NVIDIA-Merlin/systems).

2. Install required documentation tools and extensions:

   ```sh
   cd models
   pip install -r requirements-dev.txt
   ```

3. Navigate to `models/docs/` and transform the documentation to HTML output:

   ```sh
   make html
   ```

   This should run Sphinx in your shell, and output HTML in
   `build/html/index.html`

## Preview the documentation build

1. To view the docs build, run the following command from the `build/html`
   directory:

   ```sh
   python -m http.server -d build/html
   ```

