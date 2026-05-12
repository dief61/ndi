python3 - << 'EOF'
import re
path = "/home/mdi/reg-mo/ndi/services/ingest/chunker_config.yaml"
content = open(path, "rb").read()
cleaned = re.sub(rb'[\x00-\x07\x08\x0b\x0c\x0e-\x1f]', b'', content)
open(path, "wb").write(cleaned)
print(f"Bereinigt: {len(content) - len(cleaned)} Zeichen entfernt")
EOF