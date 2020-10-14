from .constants import CONDA_DEFAULT_ENVIRON


DICT_OF_CONDA_ENVIRONS = {
  CONDA_DEFAULT_ENVIRON: {
    "pythonVersion": "2.7",
    "dependencies": ["scikit-image=0.14.2"],
    "channels": ["anaconda"],
    "pipPackages": [],
    "defaultInstallOptions": {},  # Tags to be replaced to %(tag)s
  },

  # "micrograph_cleaner_em": {
  #   "pythonVersion": "3.6",
  #   "dependencies": ["numpy=1.16.4", "micrograph-cleaner-em"],
  #   "channels": ["rsanchez1369", "anaconda", "conda-forge"],
  #   "pipPackages": [],
  #   "defaultInstallOptions": {},
  # }

}
