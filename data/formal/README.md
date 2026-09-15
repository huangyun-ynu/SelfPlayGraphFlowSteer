# Formal dataset package

The seven-dataset training pool is stored as an AES-256-CBC encrypted archive in
`private/formal_data.tar.gz.enc`. The decryption secret is intentionally absent
from Git and must be restored as `SPGFS_DATA_ARCHIVE_KEY` in `.env`.

The encrypted archive contains the validated 3,584-row ADS pool and its
validation manifest. Encryption is required because HealthBench Professional
asks users not to publish benchmark examples in plain text. Run
`scripts/formal/bootstrap_remote.sh` after restoring `.env` to verify and unpack
the archive into `state/formal-data`.

Public evaluation subsets and split manifests that are safe to distribute are
under `eval/`. Stateful WebShop, ALFWorld and SWE execution assets, model
weights, API credentials and SSH identities are not Git data and must be
provisioned separately.
