python3 -c "
import yaml
with open('chunker_config.yaml') as f:
    data = yaml.safe_load(f)
print('YAML OK')
print('Sektionen:', list(data.keys()))
"