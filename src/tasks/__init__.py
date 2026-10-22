import os

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

if os.environ.get("PHP_ONLY") == "1":
    import_packages(f"{__name__}.php", _BLACKLIST_PKGS)
else:
    import_packages(__name__, _BLACKLIST_PKGS)
