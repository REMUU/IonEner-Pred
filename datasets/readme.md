SMILES structure of molecules, properties and generated descriptors are stored inside these folders.
The full_set folder contains the full dataset for all three tasks.
Those descriptor_set folders contain dataset as the combination of descriptors with respect to tasks.

The structure of Dataset is:
For NIST IE datasets:
|| CAS Name || InChI || CAS Link || smiles || IE || Descriptors || ... || ...
For Freesolv and Lipophilicity datasets:
|| id || smiles || the property || Descriptors || ... || ...