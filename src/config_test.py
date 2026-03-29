# smoke-test: verifies the config file is read correctly and paths are resolved
from FlowDetection.config import load_config


def main():
    # load_config resolves relative paths from this script's directory automatically
    config = load_config("config_test_config.yaml")
    print(config)


if __name__ == "__main__":
    main()
