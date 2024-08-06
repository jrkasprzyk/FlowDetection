from flow_detection.src.config import create_config


def main():
    config = create_config("config_test_config.yaml")
    print(config)


if __name__ == "__main__":
    main()
