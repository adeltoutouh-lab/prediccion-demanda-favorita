"""Descargar la copia pública usada en esta ejecución y verificar sus hashes.

Alternativa: descargar los siete CSV desde Kaggle y copiarlos en data/raw.
Los datos originales no se publican en este repositorio.
"""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/source_manifest.json").read_text())
    folder = root / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    expected = {item["name"]: item for item in manifest["files"]}
    if all((folder / name).is_file() and hashlib.sha256((folder / name).read_bytes()).hexdigest() == info["sha256"]
           for name, info in expected.items()):
        print("Los siete CSV ya están descargados y verificados.")
    else:
        print("Descargando copia pública identificada en source_manifest.json...")
        with urllib.request.urlopen(manifest["download_source"], timeout=60) as response:
            content = response.read(50_000_001)
        if len(content) > 50_000_000:
            raise ValueError("El archivo descargado supera el tamaño previsto")
        if hashlib.sha256(content).hexdigest() != manifest["archive_sha256"]:
            raise ValueError("El hash del ZIP no coincide con la ejecución documentada")
        extracted = set()
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for info in archive.infolist():
                name = Path(info.filename).name
                if name not in expected:
                    continue
                if name in extracted or info.file_size != expected[name]["bytes"]:
                    raise ValueError(f"Entrada inesperada: {name}")
                data = archive.read(info)
                if hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
                    raise ValueError(f"Hash incorrecto: {name}")
                (folder / name).write_bytes(data)
                extracted.add(name)
        if extracted != set(expected):
            raise ValueError("Faltan archivos en el ZIP")
        print("Datos descargados y verificados.")
