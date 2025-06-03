from pathlib import Path

import yaml
import sys


def run(base_yaml_filename: str, dump_input=True):
    with open(base_yaml_filename, 'r') as efile:
        gen_yaml = yaml.safe_load(efile)

        if dump_input:
            # pprint.PrettyPrinter().pprint(swaps)
            print(yaml.dump(gen_yaml, sort_keys=False))
            print('-------------------')

        # load parameters from the yaml
        # version = gen_yaml['metadata']['version']
        models_dir = gen_yaml['metadata']['models_dir']
        swap_dir = gen_yaml['metadata']['swap_dir']

        for gguf in gen_yaml['all-gguf'].items():
            swap_name = gguf[0]
            swap_dict = gguf[1]
            with open(f'{swap_dir}/{swap_name}.yml', 'w') as ofile:

                # write the stuff for all yamls
                ofile.write(yaml.dump({'metadata': gen_yaml['metadata']}, sort_keys=False))
                ofile.write('\n')
                ofile.write(yaml.dump(gen_yaml['all-yml'], sort_keys=False))
                ofile.write('\n')
                ofile.write('models:\n')

                for filepath in Path(models_dir).glob('*.gguf'):
                    if filepath.is_file():
                        model_name = filepath.stem.lower()
                        ofile.write(f'  "{model_name}":\n')
                        ofile.write(f'    proxy: {swap_dict['proxy']}\n')

                        # check for and handle custom settings
                        custom_cmd = {}
                        custom_model = gen_yaml['custom-gguf'].get(model_name)
                        if custom_model is not None and swap_name in custom_model.get('swaps'):
                            custom_cmd = custom_model.get('cmd')

                        ofile.write(f'    cmd: >\n')
                        for cmd_part in swap_dict['cmd'].items():
                            if cmd_part[0] in custom_cmd.keys():
                                ofile.write(f'      {custom_cmd[cmd_part[0]]}\n')
                            else:
                                ofile.write(f'      {cmd_part[1]}\n')
                        ofile.write(f'      -m {filepath.as_posix()}\n')


if __name__ == "__main__":
    gen_filename = sys.argv[1] # 'llama-swap-generate.yml'
    run(gen_filename, dump_input=True)
