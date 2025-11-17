from src.registry import load_registry_from_yaml

def main():
    reg = load_registry_from_yaml('../configs/datasets.yaml')
    print('Registered datasets:')
    for name in reg.list():
        print(" -", name)

if __name__ == '__main__':
    main()
