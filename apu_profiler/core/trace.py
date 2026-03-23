import json


def save_trace(profile, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)
