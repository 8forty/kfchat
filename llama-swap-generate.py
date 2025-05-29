import pprint
from pathlib import Path

import yaml


def run(base_yaml_filename: str, models_dir: str, dump_input=True):
    with open(base_yaml_filename, 'r') as efile:
        gen_yaml = yaml.safe_load(efile)

        if dump_input:
            # pprint.PrettyPrinter().pprint(swaps)
            print(yaml.dump(gen_yaml, sort_keys=False))
            print('-------------------')

        # print the stuff for all yamls
        print(yaml.dump({'metadata': gen_yaml['metadata']}, sort_keys=False))
        print(yaml.dump(gen_yaml['all-yml'], sort_keys=False))

        # models
        print('models:')
        for filepath in Path(models_dir).glob('*.gguf'):
            for gguf in gen_yaml['all-gguf'].items():
                print(f'  "{filepath.stem}.{gguf[0]}":')
                print(f'    proxy: {gguf[1]['proxy']}')
                print(f'    cmd: >')
                for cmd_part in gguf[1]['cmd'].items():
                    print(f'      {cmd_part[1]}')
                print(f'      -m {str(filepath)}')


if __name__ == "__main__":
    model_dir = 'z:/ggufs'
    gen_filename = 'llama-swap-generate.yml'
    run(gen_filename, model_dir, dump_input=False)
