from pathlib import Path

from cryoPARES.configManager.config_builder import build_config_structure

configs_root = Path(__file__).parent.parent / "configs"

# Build the config structure

MainConfig = build_config_structure(configs_root,
                                    dict(
                                        cachedir=(Path, "/tmp/cache")
                                    )
                                    )

# Create an instance
main_config = MainConfig()
print(main_config)
print()