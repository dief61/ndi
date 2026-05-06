bash# Alle Dokumente aus allen Buckets löschen:
docker exec -it mnr-minio mc alias set local http://localhost:9000 mnr_admin mnr_minio_password

docker exec -it mnr-minio mc rm --recursive --force local/mnr-dokumente/
docker exec -it mnr-minio mc rm --recursive --force local/mnr-artefakte/
docker exec -it mnr-minio mc rm --recursive --force local/mnr-informationsmodelle/

# Prüfen ob alles leer ist:
docker exec -it mnr-minio mc ls local/mnr-dokumente/


bashcd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate

#minIO alles Löschen - Python Variante
python3 - << 'EOF'
import sys
sys.path.insert(0, ".")
from app.services.storage import DocumentStorage

storage = DocumentStorage()
minio   = storage._get_minio()

for bucket in ["mnr-dokumente", "mnr-artefakte", "mnr-informationsmodelle"]:
    objects = list(minio.list_objects(bucket, recursive=True))
    for obj in objects:
        minio.remove_object(bucket, obj.object_name)
    print(f"  {bucket}: {len(objects)} Objekte gelöscht")
EOF
