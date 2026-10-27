"""Execute the generated notebook and save actual IOPub outputs after every cell.

Set M3_RESOURCE_DIR, M3_CHECKPOINT_DIR, M3_LPIPS_CACHE and M3_MODEL_DIR.
M3_PYTHON optionally identifies the archived ML Python environment.
"""
import argparse
import datetime
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def execute(notebook: Path, output: Path):
    nb = nbformat.read(notebook, as_version=4)
    # Resolve the default ipykernel executable to this isolated notebook environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    destination = output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def persist(cell, cell_index, **kwargs):
        nbformat.write(nb, destination)
        kind = cell.cell_type
        if kind == "code":
            errors = [o for o in cell.get("outputs",[]) if o.output_type == "error"]
            now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"{now} cell {cell_index+1}/{len(nb.cells)} executed; outputs={len(cell.get('outputs',[]))}; errors={len(errors)}", flush=True)

    client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False,
                            record_timing=True, resources={"metadata":{"path":str(destination.parent)}},
                            on_cell_executed=persist)
    try:
        client.execute()
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None and cell.outputs, "Missing genuine output"
                assert not any(o.output_type == "error" for o in cell.outputs)
        nb.metadata["m3_scope"]["execution_outputs"] = "Genuine local Jupyter execution; see hardware cell. Colab CUDA not yet executed."
        nbformat.validate(nb)
        nbformat.write(nb,destination)
        html,_ = HTMLExporter().from_notebook_node(nb)
        destination.with_suffix(".html").write_text(html)
        print("COMPLETE:", destination, flush=True)
    except BaseException:
        nbformat.write(nb,destination)
        print("Stopped with execution evidence preserved:",destination,flush=True)
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    execute(args.notebook,args.output)
