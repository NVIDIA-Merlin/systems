# Documentation

This folder contains the scripts necessary to build documentation. You can
find the documentation at <https://nvidia-merlin.github.io/systems/main>.

1. Follow the instructions to create a Python developer environment in
   the repository README.

2. Install required documentation tools and extensions:

   ```shell
   cd models
   pip install -r requirements/dev.txt
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

## Special Considerations for Examples and Copydirs

Freely add examples and images to the `examples` directory.
After you add a notebook or README.md file, update the
`docs/source/toc.yaml` file with entries for the new files.

Be aware that `README.md` files are renamed to `index.md`
during the build, due to the `copydirs_file_rename` setting.
If you add `examples/blah/README.md`, then add an entry in
the `toc.yaml` file for `examples/blah/index.md`.

## Special Considerations for the fork of sphinx-multiversion

So that we could use the `sphinx-external-toc` and `sphinxcontrib-copydirs`
extensions with current development and still build a few older versions
with `sphinx-multiversion`, the fork of `sphinx-multiversion` is enhanced
so that when it begins a build for a tag, it checks for a `<tag>-docs` branch.

If a ref for a `<tag>-docs` branch is found, then `sphinx-multiversion` builds
from the source in the alternative branch.
