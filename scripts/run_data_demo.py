from src.registry import load_registry_from_yaml

def main():
    reg = load_registry_from_yaml("../configs/datasets.yaml")
    for name in reg.list().keys():
        db = reg.load(name)
        print(f"{name:>25} | n={db.x.size:4d} | removed_mean={db.meta['center_meta']['removed_mean']:.6f} | y_true={db.y_true is not None}")

if __name__ == "__main__":
    main()
