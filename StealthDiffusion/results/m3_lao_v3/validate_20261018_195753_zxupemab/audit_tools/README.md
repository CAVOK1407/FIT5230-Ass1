This is the exact script used for independent_data_audit.json. It reads only fixed input and provenance files and does not load models or inspect attack scores.

Run with the fit5230m2 Python environment. The ROOT and OUT constants bind it to this validation run. It deliberately refuses to overwrite an existing audit; for an independent rerun, use a separate copy and change only OUT to a new audit filename.

Script SHA-256: c4f2f5deeb1175dc10d57bda1a612c8c7cf2a7dd40741e1e525ac087e23c7ebc
