# Documentation

This folder contains the scripts necessary to build documentation. You can
find the documentation at <https://nvidia-merlin.github.io/systems/main>.

## Contributing to Docs

You build the documentation with the `tox` command and specify the `docs` environment.
The following steps are one way of many to build the documentation before opening a merge request.

1. Create a virtual environment:

   ```shell
   python -m venv .venv
   ```

1. Activate the virtual environment:

   ```shell
   source .venv/bin/activate
   ```

1. Install tox in the virtual environment:

   ```shell
   python -m pip install --upgrade pip
   python -m pip install tox
   ```

1. Build the documentation with tox:

   ```shell
   tox -e docs
   ```

These steps run Sphinx in your shell and create HTML in the `docs/build/html/`
directory.

## Preview the Changes

View the docs web page by opening the HTML in your browser. First, navigate to
the `build/html/` directory and then run the following command:

```shell
python -m http.server
```

Afterward, open a web browser and access <https://localhost:8000>.

Check that yours edits formatted correctly and read well.

## Special Considerations for Examples and Copydirs

Freely add examples and images to the `examples` directory.
After you add a notebook or README.md file, update the
`docs/source/toc.yaml` file with entries for the new files.

Be aware that `README.md` files are renamed to `index.md`
during the build, due to the `copydirs_file_rename` setting.
If you add `examples/blah/README.md`, then add an entry in
the `toc.yaml` file for `examples/blah/index.md`.
