import json
import sys
import os


def main():
    json_path = sys.argv[1]
    new_json_path = sys.argv[2]


    with open(json_path) as fh:
        data = json.load(fh)

    name = data['name']

    if '_' in name:
        name_list = name.split('_')
        old_id = int(name_list[-1])
        new_id = old_id + 1
        name_list[-1] = str(new_id)
        data['name'] = '_'.join(name_list)

        dir_name = os.path.dirname(new_json_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(new_json_path, 'w') as fh:
            json.dump(data, fh)

if __name__ == "__main__":
    main()