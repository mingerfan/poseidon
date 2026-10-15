"""Maintainer helper: regenerate package file hashes after reviewed changes."""
import json
from prepare_assets import PACKAGE,digest
files={file.relative_to(PACKAGE).as_posix():digest(file) for file in sorted(PACKAGE.rglob('*'))
       if file.is_file() and file.name!='manifest.sha256.json'
       and '__pycache__' not in file.parts
       and not any(part.startswith('build') and '.' not in part for part in file.relative_to(PACKAGE).parts)}
(PACKAGE/'manifest.sha256.json').write_text(json.dumps(files,indent=2)+'\n')
print(len(files),'package files hashed')
