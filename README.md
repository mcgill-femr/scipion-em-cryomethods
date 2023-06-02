# Cryomethods plugin for Scipion

Cryomethods is a cryo-electron microscopy image processing plugin of Scipion framwork focussed mainly on processing macromolecular complexes showing extensive heterogeneity. Cryomethods will be integrated in the Scipion plugin manager but in the meantime you can install from the following steps:

## Installation

### Docker (recommended)

This is the easiest and most reliable way to install and run Cryomethods if you don't have scipion previously installed on your computer.

1. Install Docker (you don't have it already)
To install Docker on your machine check the instructions on the official webpage https://docs.docker.com/get-docker/

2. Download the Dockerfile provided [here](./Docker/Dockerfile)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
