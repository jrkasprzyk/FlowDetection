import os

from FlowDetection.config import create_config

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"script_dir: {script_dir}")

# this is a test script to check that the config file is being read correctly and that the paths are being set correctly
def main():
    config = create_config(os.path.join(script_dir, "config_test_config.yaml"))
    print(config)


if __name__ == "__main__":
    main()
